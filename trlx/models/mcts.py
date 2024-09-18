import torch
import torch.nn.functional as F
import numpy as np
import math
from trlx.models.modeling_ilql import AutoModelForCausalLMWithILQLHeads

class MCTSNode:
    def __init__(self, state, model, parent=None, action=None):
        self.state = state  # Contains input_ids, attention_mask, position_ids, past_key_values
        self.model = model  # Reference to the model to perform forward passes
        self.parent = parent
        self.action = action  # The token that led to this state
        self.children = {}  # action -> child node
        self.N = 0  # Total visit count for this node
        self.N_sa = None  # Visit counts for state-action pairs
        self.W_sa = None  # Total value for state-action pairs
        self.Q_sa = None  # Mean value for state-action pairs
        self.P_sa = None  # Prior probabilities for actions at this node
        self.is_terminal = False  # Flag to indicate if this node is terminal
        self.value = None  # Cached value of the node

    def expand_and_evaluate(self, eos_token_id, logit_mask=None):
        if self.is_terminal:
            return 0

        input_ids = self.state['input_ids']
        attention_mask = self.state['attention_mask']
        position_ids = self.state['position_ids']
        past_key_values = self.state['past_key_values']

        with torch.inference_mode():
            out = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                bypass_peft_prompt_adapter=False,
            )

            logits, _, target_qs, vs, _ = out

            if self.model.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[0][:, -1, :]

            logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            vs = vs[:, -1, :]  # Shape: [batch_size, 1]

            adv = qs - vs  # Shape: [batch_size, vocab_size]
            pi_beta = F.log_softmax(logits, dim=-1)  # Shape: [batch_size, vocab_size]

            adjusted_logits = pi_beta + self.model.alpha * adv  # Using alpha as beta

            if logit_mask is not None:
                adjusted_logits += logit_mask  # Assuming logit_mask is a tensor of -inf for masked tokens

            # Compute prior probabilities P(s,a)
            P_sa = F.softmax(adjusted_logits / self.model.temperature, dim=-1).squeeze(0).cpu().numpy()
            self.P_sa = P_sa  # Shape: [vocab_size]

            # Initialize visit counts and values for each action
            vocab_size = self.model.base_model.config.vocab_size
            self.N_sa = np.zeros(vocab_size, dtype=np.int32)
            self.W_sa = np.zeros(vocab_size, dtype=np.float32)
            self.Q_sa = np.zeros(vocab_size, dtype=np.float32)

            # Check for terminal state
            if input_ids[0, -1].item() == eos_token_id:
                self.is_terminal = True
                self.value = 0  # Terminal nodes have zero value
                return self.value

            # Estimate value using vs
            self.value = vs.squeeze(0).item()
            return self.value

    def select_action(self, c_puct):
        # Use the PUCT formula to select the next action
        sqrt_N = math.sqrt(self.N + 1e-8)
        U = self.Q_sa + c_puct * self.P_sa * sqrt_N / (1 + self.N_sa)
        best_action = np.argmax(U)
        return best_action

    def backup(self, value):
        # Backpropagate the value up the tree
        self.N += 1
        if self.parent is not None:
            a = self.action
            self.parent.N_sa[a] += 1
            self.parent.W_sa[a] += value
            self.parent.Q_sa[a] = self.parent.W_sa[a] / self.parent.N_sa[a]
            self.parent.backup(value)

class Peach(AutoModelForCausalLMWithILQLHeads):
    # Modify the generate method to use MCTS
    def generate(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        beta=1,
        max_new_tokens=32,
        max_length=1024,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=None,
        eos_token_id=None,
        num_simulations=50,
        c_puct=1.0,
    ):
        """
        Generates samples using token-level Monte Carlo Tree Search guided by qs and vs.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.base_model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.base_model.config.eos_token_id
        self.temperature = temperature  # Store temperature in the model for use in adjusted logits
        self.alpha = beta  # Use beta as alpha in adjusted logits

        if attention_mask is None:
            attention_mask = input_ids.ne(pad_token_id).long()

        if position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)

        samples = input_ids.clone()
        max_new_tokens = min(max_new_tokens, max_length - input_ids.shape[1])

        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        bypass_peft = False

        for _ in range(max_new_tokens):
            if finished.all():
                break

            # Initialize the root node
            root_state = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'past_key_values': past_key_values,
            }
            root_node = MCTSNode(root_state, self)
            root_node.expand_and_evaluate(eos_token_id, logit_mask=logit_mask)

            # Perform MCTS simulations
            for _ in range(num_simulations):
                node = root_node
                search_path = [node]

                while not node.is_terminal:
                    if node.N == 0:
                        # Expand and evaluate leaf node
                        value = node.expand_and_evaluate(eos_token_id, logit_mask=logit_mask)
                        break
                    else:
                        # Select action
                        action = node.select_action(c_puct)
                        if action in node.children:
                            node = node.children[action]
                            search_path.append(node)
                            continue

                        # Create a new child node
                        action_tensor = torch.tensor([[action]], device=input_ids.device)
                        next_state = {
                            'input_ids': torch.cat([node.state['input_ids'], action_tensor], dim=1),
                            'attention_mask': torch.cat([node.state['attention_mask'], torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)], dim=1),
                            'position_ids': node.state['position_ids'] + 1,
                            'past_key_values': node.state['past_key_values'],  # Not updating past_key_values to save memory
                        }
                        child_node = MCTSNode(next_state, self, parent=node, action=action)
                        value = child_node.expand_and_evaluate(eos_token_id, logit_mask=logit_mask)
                        node.children[action] = child_node
                        search_path.append(child_node)
                        break  # Stop at new leaf node

                # Backup value
                value = node.value if node.value is not None else 0
                for node in reversed(search_path):
                    node.backup(value)

            # Choose the action with the highest visit count from the root node
            best_action = np.argmax(root_node.N_sa)
            action_tensor = torch.tensor([[best_action]], device=input_ids.device)

            # Update the state with the chosen action
            input_ids = torch.cat([input_ids, action_tensor], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)], dim=1)
            position_ids = position_ids + 1  # Assuming position_ids increments by 1

            # Update past_key_values by performing a forward pass
            with torch.inference_mode():
                out = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    bypass_peft_prompt_adapter=bypass_peft,
                )
                past_key_values = out.past_key_values  # Update past_key_values for next iteration

            samples = torch.cat([samples, action_tensor], dim=1)
            finished |= action_tensor.squeeze(1) == eos_token_id

            # Handle PEFT models
            if self.peft_type and _ == 0 and "LORA" not in self.peft_type:
                bypass_peft = True
                prefix_length = self.peft_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(batch_size, prefix_length, device=input_ids.device, dtype=torch.long)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        return samples