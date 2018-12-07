"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

import onmt
from onmt.modules.position_ffn import PositionwiseFeedForward

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn_type="scaled-dot"):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn_type = self_attn_type

        if self_attn_type == "scaled-dot":
            self.self_attn = onmt.modules.MultiHeadedAttention(
                heads, d_model, dropout=dropout)
        elif self_attn_type == "average":
            self.self_attn = onmt.modules.AverageAttention(
                d_model, dropout=dropout)

        self.context_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)

        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        if self.self_attn_type == "scaled-dot":
            query, attn = self.self_attn(all_input, all_input, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")
        elif self.self_attn_type == "average":
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, all_input

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask



class MemoryLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff, dropout):
        super(MemoryLayer, self).__init__()

        self.ma_l1 = onmt.modules.MultiHeadedAttention(
                heads, d_model, dropout=dropout)
        self.ma_l2 = onmt.modules.MultiHeadedAttention(
                heads, d_model, dropout=dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ma_l1_prenorm = nn.LayerNorm(d_model, eps=1e-6)
        self.ma_l2_prenorm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)
        self.v = nn.Linear(d_model*2, d_model)
        self.w = nn.Linear(d_model, d_model)
        self.u = nn.Linear(d_model, d_model)
        self.s = nn.Sigmoid()

    def forward(self, output, src, src_m, tgt_m, src_memory_bank, sc_bias, s_bias):
        # self multihead attention
        batch, lens, dim = src_memory_bank.size()

        src_memory_bank = src_memory_bank.view(batch*lens, 1, dim)
        outputs_m = src_m.view(batch, lens, 7, dim).view(batch*lens, 7, dim)
        s_norm_x = self.ma_l1_prenorm(src_memory_bank)
        s_y, s_ = self.ma_l1(outputs_m, outputs_m, s_norm_x,
                            mask=sc_bias,
                            layer_cache=None,
                            type="context")

        s_x = self.dropout(s_y)+src_memory_bank
        s_x = s_x.view(batch, lens, dim)

        outputt_m = tgt_m
        s_x = s_x.unsqueeze(2).repeat(1, 1, 2, 1).view(batch, lens*2, dim)

        s_t_m = torch.cat((outputt_m, s_x), dim=2)
        s_t_m = self.v(s_t_m)

        s_t_norm_x = self.ma_l2_prenorm(output)
        s_t_y, s_t_ = self.ma_l2(s_t_m, s_t_m, s_t_norm_x,
                            mask=s_bias,
                            layer_cache=None,
                            type="context")

        output_h = self.ffn(self.dropout(s_t_y))

        B = self.s(self.w(output) + self.u(output_h))
        ans = (1 - B) * output + B * output_h

        return ans, s_t_, B


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, attn_type,
                 copy_attn, self_attn_type, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.self_attn_type = self_attn_type

        # Decoder State
        self.state = {}

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             self_attn_type=self_attn_type)
             for _ in range(num_layers)])
        self.memory = MemoryLayer(d_model, heads, d_ff, dropout)

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                d_model, attn_type=attn_type)
            self._copy = True
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def init_state(self, src, src_m, tgt_m, with_cache=False):
        """ Init decoder state """
        self.state["src"] = src
        self.state["src_m"] = src_m
        self.state["tgt_m"] = tgt_m
        self.state["previous_input"] = None
        self.state["previous_layer_inputs"] = None
        self.state["cache"] = None


    def update_state(self, new_input, previous_layer_inputs):

        self.state["previous_input"] = new_input
        self.state["previous_layer_inputs"] = previous_layer_inputs

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["previous_input"] is not None:
            self.state["previous_input"] = fn(self.state["previous_input"], 1)
        if self.state["previous_layer_inputs"] is not None:
            self.state["previous_layer_inputs"] = \
                fn(self.state["previous_layer_inputs"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        if self.state["previous_input"] is not None:
            self.state["previous_input"] = \
                self.state["previous_input"].detach()
        if self.state["previous_layer_inputs"] is not None:
            self.state["previous_layer_inputs"] = \
                self.state["previous_layer_inputs"].detach()
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, emb_src, emb_srcm, memory_bank, memory_lengths=None,
                step=None, cache=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        src = self.state["src"]
        src_m = self.state["src_m"]
        tgt_m = self.state["tgt_m"]
        src_len, src_batch, _ = src.size()
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_m_words = src_m[:, :, 0].transpose(0, 1)
        tgt_m_words = tgt_m[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        tgt_m = tgt_m.transpose(0, 1).contiguous().view(src_batch, src_len, 2, 1).transpose(0, 1).contiguous().view(src_len, src_batch * 2, 1)

        # Initialize return variables.
        dec_outs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt, step=step)
        emb_m = self.embeddings(tgt_m)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        outputs = emb_src.transpose(0, 1).contiguous()
        outputs_m = emb_srcm
        outputt_m = emb_m.view(src_len, src_batch, 2, -1).transpose(0, 1).contiguous().view(src_batch, src_len * 2, -1)
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        srcm_pad_mask = src_m_words.data.eq(padding_idx).view(src_batch, src_len, 7).view(src_batch*src_len, 7).unsqueeze(1) \
            .expand(src_batch*src_len, 1, 7)
        tgtm_pad_mask = tgt_m_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len*2)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if self.state["cache"] is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if self.state["cache"] is None:
                if self.state["previous_input"] is not None:
                    prev_layer_input = self.state["previous_layer_inputs"][i]
            output, attn, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=self.state["cache"]["layer_{}".format(i)]
                    if self.state["cache"] is not None else None,
                    step=step)
            if self.state["cache"] is None:
                saved_inputs.append(all_input)

        output, attn, B = self.memory(output, outputs, outputs_m, outputt_m, src_memory_bank, srcm_pad_mask, tgtm_pad_mask)

        if self.state["cache"] is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        dec_outs = output.transpose(0, 1).contiguous()
        B = B.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        if self.state["cache"] is None:
            self.update_state(tgt, saved_inputs)
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns, B

    def _init_cache(self, memory_bank, num_layers, self_attn_type):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            if self_attn_type == "scaled-dot":
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            elif self_attn_type == "average":
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth))
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(l)] = layer_cache
