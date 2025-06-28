import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Iterable
from dataclasses import dataclass
from CONSTANTS import *


t = torch.tensor(list(range(5, 10)) + list(range(3, 9)), dtype=torch.float32)
print(t)


vocab_size = 5
embed_size = 7
batch_size = 3
byte_embedding = nn.Embedding(vocab_size, embed_size)

# Now fabricate some data with shape [batch_size, length]
test_t = torch.tensor(np.random.randint(0, 5, (batch_size, 8)), dtype=torch.long)
t2 = torch.zeros((batch_size, 3), dtype=torch.long)
print(t2)

t = torch.tensor(list(range(10)))
for j in range(len(t)):
    # Now get a context window and pad with null characters
    # Any target payload that we get passed will begin with the SOS character
    # Always start at j and fill the rest with nulls
    byte_context = torch.ones(10, dtype=torch.long) * NULL
    byte_context[max(10 - j - 1, 0) :] = t[max(j + 1 - 10, 0) : j + 1]

    print(byte_context)

print(list(range(12))[10:])


# class NextPacketPredictor(nn.Module):
#     """
#     @Description: We take a context history of past packets, the target packet's meta data
#     and the past bytes if there are any to autoregressively predict the next byte in the
#     sequence.
#         Input for next byte: CAT_EMBED_DIMS + NUM_EMBED_DIMS + CONVERSATION_CONTEXT_DIMS + (SEQ_LEN * BYTE_EMBED_DIMS)

#     Since the byte context size is fixed, but we are allowing for variable length input
#     sequences we need to use the NULL token for padding to the FRONT!!. In the sequence the
#     byte at the very end is considered the most recent one.

#     @Notes:
#         - All the input except for the sequence embeddings remains the same.
#         - We will not enforce a fixed sequence length, but rather pack padded sequences
#         - In addition to the 256 byte vocabulary we also enlist a number of special tokens
#         each defined in the CONSTANTS.py file.
#         - The SOS characeter at the start of each packet counts in the sequence length count

#     @Returns:
#     """

#     def __init__(self, categorical_dims=list(), numerical_dim=0):
#         super().__init__()

#         # Encode next packet metadata
#         cat_embed_dim = sum(max(50, cat_size // 2) for cat_size in categorical_dims)
#         self.cat_embeddings = nn.ModuleList(
#             [
#                 nn.Embedding(cat_size, max(50, cat_size // 2))
#                 for cat_size in categorical_dims
#             ]
#         )

#         # Combine conversation context with next packet metadata
#         context_dim = (
#             CONVERSATIONAL_HIDDEN_DIM
#             + cat_embed_dim
#             + numerical_dim
#             + (BYTE_CONTEXT_LEN * BYTE_EMBED_DIM)
#         )

#         # Decoder LSTM for payload generation
#         self.decoder_lstm = nn.LSTM(
#             input_size=context_dim,
#             hidden_size=CONVERSATIONAL_HIDDEN_DIM,
#             num_layers=NEXT_PACKET_LAYERS,
#             batch_first=True,
#             dropout=NEXT_PACKET_DROPOUT,
#         )

#         # Output projection
#         self.output_projection = nn.Linear(CONVERSATIONAL_HIDDEN_DIM, BYTE_VOCAB_DIM)

#         # Byte embedding for decoder
#         self.byte_embedding = nn.Embedding(BYTE_VOCAB_DIM, BYTE_EMBED_DIM)

#     def forward(
#         self,
#         embedded_conversation_context: torch.Tensor,  # embedded context
#         next_packet_categorical: torch.Tensor,
#         next_packet_numerical: torch.Tensor,
#         target_payload: None | torch.Tensor = None,
#         attn_mask: None | torch.Tensor = None,
#     ):
#         """
#         @Args:
#             conversation_context: [batch, conversation_hidden_dim]
#             next_packet_categorical: [batch, num_categorical]
#             next_packet_numerical: [batch, numerical_dim]
#             target_payload: [batch, payload_length] - for teacher forcing during training
#         @Notes:
#             - The mqtt.len is the third arguement in the numerical features
#         """
#         batch_size = embedded_conversation_context.shape[0]

#         # Embed next packet categorical features
#         cat_embeds = []
#         for i, embedding_layer in enumerate(self.cat_embeddings):
#             cat_embeds.append(embedding_layer(next_packet_categorical[:, i]))

#         cat_embeds = (
#             torch.cat(cat_embeds, dim=-1) if cat_embeds else torch.empty(batch_size, 0)
#         )

#         # Combine context with next packet metadata as predictors
#         context = torch.cat(
#             [embedded_conversation_context, cat_embeds, next_packet_numerical], dim=-1
#         )

#         if target_payload is None:
#             # Get the mqtt length as the third argument in the numerical features
#             lengths = next_packet_numerical[:, 2]
#             # Inference mode - autoregressive generation
#             return self._generate_payload(context, lengths)
#         else:
#             assert not isinstance(
#                 attn_mask, type(None)
#             ), f"An attention mask corresponding to the target payload must be provided"

#             assert (
#                 target_payload.shape == attn_mask.shape
#             ), f"Target payload shape: {target_payload.shape} != attention mask shape: {attn_mask.shape}"

#             # Training mode - teacher forcing
#             return self._train_forward(context, target_payload, attn_mask)

#     def _train_forward(
#         self, context, target_payload: torch.Tensor, attn_mask: torch.Tensor
#     ):
#         """
#         @Description: Uses the batch contexts along the prior embedded next packet features
#         to predict the entire sequence of next bytes for the target packet.

#         @Notes:
#             context: [batch_size, context_length, packet_embed_size]
#             target_payload: [batch_size, payload_length]

#         @TODO:
#             - Should I use the mqtt.length to explicitly cut the prediciton off or rather
#             allow the model to learn that it should stop at mqtt.length

#         @Returns:
#             logits: torch.tensor([]) -> [batch_size, payload_length, BYTE_VOCAB_DIMS]
#         """
#         batch_size, max_len = target_payload.shape  # [batch_size, payload_length]

#         # Since we are dealing with variable payload lengths we will need to pack the payloads
#         # This operates on the assumption that
#         payload_lens = attn_mask.sum(dim=1)

#         # Embed decoder input
#         decoder_embeds = self.byte_embedding(
#             target_payload
#         )  # [batch_size, max_len, BYTE_EMBED_DIMS]

#         # ensure that the attention mask is applied to the embeddings
#         mask = attn_mask.unsqueeze(-1)
#         mask_embeds = decoder_embeds * mask

#         # Repeat context for each timestep in the payload so that we predict byte by byte
#         context_repeated = context.unsqueeze(1).repeat(1, max_len, 1)

#         # Combine context with decoder embeddings
#         lstm_input = torch.cat([context_repeated, mask_embeds], dim=-1)
#         packed_input = nn.utils.rnn.pack_padded_sequence(
#             lstm_input, payload_lens.cpu(), batch_first=True, enforce_sorted=False
#         )

#         # Process through decoder LSTM and unpad
#         packed_output, _ = self.decoder_lstm(packed_input)
#         decoder_output, _ = nn.utils.rnn.pad_packed_sequence(
#             packed_output, batch_first=True
#         )

#         # Project to vocabulary
#         logits = self.output_projection(decoder_output)

#         return logits  # [batch_size, payload_size, BYTE_EMBED_DIMS]

#     def _generate_payload(
#         self, context: torch.Tensor, msg_lens: torch.Tensor
#     ) -> List[SeqInput]:
#         """
#         @Description: Goes through autoregressively predicting the packet payload byte by byte

#         @Notes:
#             - The length of the payload must be provided. Typically this is contained within
#             the meta data.

#         @TEST:
#             - Does this padded sequnce only predict for a single byte? What is the output shape?

#         @Returns: List[SeqInput]
#         """
#         # Autoregressively predict the next byte in the packet, adding it to the context each time
#         ctx_batch_size, context_len = context.shape
#         batch_size = msg_lens.shape[0]
#         assert (
#             ctx_batch_size == batch_size
#         ), f"_generate_payload failed with batch size mismatch {batch_size} != {ctx_batch_size}"

#         max_len = int(msg_lens.max())
#         pred_payloads = torch.zeros((batch_size, max_len), dtype=torch.uint8)

#         for i in range(max_len):
#             # Encode only the prior bytes
#             byte_embeds = self.byte_embedding(pred_payloads[:, :i])

#             # make a copy of the context for the current timestamp
#             context_step = context.unsqueeze(1)

#             if i > 1:
#                 lstm_input = torch.cat([context_step, byte_embeds], dim=1)
#             else:
#                 lstm_input = context_step

#             # Get the LSTM output for the given input
#             output, _ = self.decoder_lstm(lstm_input)

#             logits = self.output_projection(
#                 output[:, -1, :]
#             )  # Only gets the very last prediction
#             # Select the most likely byte for each
#             pred_byte = logits.argmax(dim=-1)

#             pred_payloads[:, i] = pred_byte

#         # Now create attention mask for each msg length
#         attn_masks = torch.tensor(
#             [[1] * s_len + [0] * (max_len - s_len) for s_len in msg_lens],
#             dtype=torch.bool,
#         )

#         # Now construct the sequence of SeqInputs
#         return [
#             SeqInput(payload, attn_mask, int(attn_mask.sum(dim=1)))
#             for payload, attn_mask in zip(pred_payloads, attn_masks)
#         ]
