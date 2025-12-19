import torch
from torch import nn

import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'


class CheckInEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# Attention for query and key with different dimension
class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query)  # [embed_size]
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out


class CLSPRec(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=60,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5

        # Layers
        self.embedding = CheckInEmbedding(
            f_embed_size,
            vocab_size
        )
        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_lstm_layers,
            dropout=0
        )
        self.final_attention = Attention(
            qdim=f_embed_size,
            kdim=self.total_embed_size
        )
        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))

        self.loss_func = nn.CrossEntropyLoss()

        self.tryone_line2 = nn.Linear(self.total_embed_size, f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))

        self.gate_mlp = nn.Sequential(
            nn.Linear(5, f_embed_size),
            nn.ReLU(),
            nn.Linear(f_embed_size, 1),
        )
        cat_vocab_size = int(vocab_size["cat"]) if isinstance(vocab_size["cat"], torch.Tensor) else vocab_size["cat"]
        self.cat_head = nn.Linear(self.total_embed_size, cat_vocab_size)

    def feature_mask(self, sequences, mask_prop):
        masked_sequences = []
        for seq in sequences:  # each long term sequences
            feature_seq, day_nums = seq[0], seq[1]
            seq_len = len(feature_seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            feature_seq[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_seq[4, masked_index] = self.vocab_size["day"]  # mask day

            masked_sequences.append((feature_seq, day_nums))
        return masked_sequences

    def ssl(self, embedding_1, embedding_2, neg_embedding):
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))

        def single_infoNCE_loss_simple(embedding1, embedding2, neg_embedding):
            pos = score(embedding1, embedding2)
            neg1 = score(embedding1, neg_embedding)
            neg2 = score(embedding2, neg_embedding)
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss

        ssl_loss = single_infoNCE_loss_simple(embedding_1, embedding_2, neg_embedding)
        return ssl_loss

    def _split_sample(self, sample):
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        return long_term_sequences, short_term_sequence

    def _unpack_sequence(self, sequence):
        if len(sequence) == 4:
            return sequence[0], sequence[1], sequence[2], sequence[3]
        if len(sequence) == 3:
            return sequence[0], sequence[1], sequence[2], None
        return sequence[0], sequence[1], None, None

    def _compute_context_gate(self, z_long, z_short, context):
        if context is None:
            alpha = torch.tensor([0.0], device=z_long.device)
            return z_short, alpha
        context = context.unsqueeze(0) if context.dim() == 1 else context
        alpha = torch.sigmoid(self.gate_mlp(context)).squeeze(0)
        z_u = alpha * z_short + (1 - alpha) * z_long
        return z_u, alpha

    def _forward_single(self, sample, neg_sample_list):
        # Process input sample
        long_term_sequences, short_term_sequence = self._split_sample(sample)
        short_term_features, _, short_context, short_label = self._unpack_sequence(short_term_sequence)
        short_term_features = short_term_features[:, :-1]
        target = short_term_sequence[0][0, -1]
        if short_label is not None:
            target_cat = short_label
        else:
            target_cat = short_term_sequence[0][1, -1]
        user_id = short_term_sequence[0][2, 0]

        # Random mask long-term sequences
        long_term_sequences = self.feature_mask(long_term_sequences, settings.mask_prop)

        # Long-term
        long_term_out = []
        for seq in long_term_sequences:
            features, _, _, _ = self._unpack_sequence(seq)
            output = self.encoder(feature_seq=features)
            long_term_out.append(output)
        long_term_catted = torch.cat(long_term_out, dim=0)

        # Short-term
        short_term_state = self.encoder(feature_seq=short_term_features)

        # User enhancement
        user_embed = self.embedding.user_embed(user_id)
        embedding = torch.unsqueeze(self.embedding(short_term_features), 0)
        output, _ = self.lstm(embedding)
        short_term_enhance = torch.squeeze(output)
        user_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.tryone_line2(
            torch.mean(short_term_enhance, dim=0))

        # SSL
        neg_short_term_states = []
        if neg_sample_list:
            for neg_day_sample in neg_sample_list:
                neg_trajectory_features = neg_day_sample[0]
                neg_short_term_state = self.encoder(feature_seq=neg_trajectory_features)
                neg_short_term_state = torch.mean(neg_short_term_state, dim=0)
                neg_short_term_states.append(neg_short_term_state)

        short_embed_mean = torch.mean(short_term_state, dim=0)
        long_embed_mean = torch.mean(long_term_catted, dim=0)
        if neg_short_term_states:
            neg_embed_mean = torch.mean(torch.stack(neg_short_term_states), dim=0)
            ssl_loss = self.ssl(short_embed_mean, long_embed_mean, neg_embed_mean)
        else:
            ssl_loss = torch.tensor(0.0, device=short_embed_mean.device)

        # Final predict
        h_all = torch.cat((short_term_state, long_term_catted))
        final_att = self.final_attention(user_embed, h_all, h_all)
        z_long = long_embed_mean
        z_short = short_embed_mean
        z_u = final_att
        alpha = None
        if settings.use_gate:
            z_u, alpha = self._compute_context_gate(z_long, z_short, short_context)
        output = self.out_linear(z_u)

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)

        pred_loss = self.loss_func(pred, label)
        loss = pred_loss + ssl_loss * settings.neg_weight
        cat_logits = self.cat_head(z_u)
        cat_label = torch.unsqueeze(target_cat, 0)
        metrics = {
            "z_u_long": z_long,
            "z_u_short": z_short,
            "z_u": z_u,
            "alpha_u": alpha,
            "cat_logits": cat_logits,
            "cat_label": cat_label,
            "loss_poi": pred_loss,
            "loss_ssl": ssl_loss,
        }
        return loss, output, metrics

    def forward_batch(self, samples, neg_sample_lists):
        batch_outputs = []
        z_longs, z_shorts = [], []
        cat_logits_list, cat_labels = [], []
        total_poi_loss = 0.0
        total_ssl_loss = 0.0
        alpha_values = []

        for idx, sample in enumerate(samples):
            neg_list = []
            if neg_sample_lists:
                neg_list = neg_sample_lists[idx]
            loss, output, metrics = self._forward_single(sample, neg_list)
            total_poi_loss = total_poi_loss + metrics["loss_poi"]
            total_ssl_loss = total_ssl_loss + metrics["loss_ssl"]
            batch_outputs.append(output)
            z_longs.append(metrics["z_u_long"])
            z_shorts.append(metrics["z_u_short"])
            cat_logits_list.append(metrics["cat_logits"])
            cat_labels.append(metrics["cat_label"])
            if metrics["alpha_u"] is not None:
                alpha_values.append(metrics["alpha_u"])

        z_longs = torch.stack(z_longs, dim=0)
        z_shorts = torch.stack(z_shorts, dim=0)
        batch_size = z_longs.shape[0]
        loss_cl = torch.tensor(0.0, device=z_longs.device)
        hard_neg_mean = torch.tensor(0.0, device=z_longs.device)

        if settings.use_contrastive and batch_size > 1:
            z_longs_norm = nn.functional.normalize(z_longs, p=2, dim=1)
            z_shorts_norm = nn.functional.normalize(z_shorts, p=2, dim=1)
            sim_matrix = torch.matmul(z_longs_norm, z_shorts_norm.t())
            pos_sim = torch.diagonal(sim_matrix)
            if settings.neg_strategy == "hard":
                mask = torch.eye(batch_size, device=sim_matrix.device).bool()
                sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))
                hard_k = min(settings.hard_k, batch_size - 1)
                hard_vals, _ = torch.topk(sim_matrix, hard_k, dim=1)
                hard_neg_mean = torch.mean(hard_vals)
                logits = torch.cat([pos_sim.unsqueeze(1), hard_vals], dim=1) / settings.tau
                loss_cl = torch.mean(-pos_sim / settings.tau + torch.logsumexp(logits, dim=1))
            else:
                mask = torch.eye(batch_size, device=sim_matrix.device).bool()
                neg_logits = sim_matrix.masked_fill(mask, float("-inf"))
                logits = torch.cat([pos_sim.unsqueeze(1), neg_logits], dim=1) / settings.tau
                loss_cl = torch.mean(-pos_sim / settings.tau + torch.logsumexp(logits, dim=1))

        loss_poi = total_poi_loss / batch_size
        loss_ssl = total_ssl_loss / batch_size
        loss_cat = torch.tensor(0.0, device=z_longs.device)
        if settings.use_aux_cat:
            cat_logits = torch.stack(cat_logits_list, dim=0)
            cat_labels = torch.stack(cat_labels, dim=0).squeeze(1)
            loss_cat = nn.functional.cross_entropy(cat_logits, cat_labels)

        total_loss = loss_poi
        if settings.use_contrastive:
            total_loss = total_loss + settings.lambda_cl * loss_cl
        elif settings.enable_ssl:
            total_loss = total_loss + loss_ssl * settings.neg_weight
        if settings.use_aux_cat:
            total_loss = total_loss + settings.lambda_cat * loss_cat

        alpha_mean = torch.mean(torch.stack(alpha_values)) if alpha_values else None
        alpha_var = torch.var(torch.stack(alpha_values)) if alpha_values else None

        metrics = {
            "loss_poi": loss_poi,
            "loss_ssl": loss_ssl,
            "loss_cl": loss_cl,
            "loss_cat": loss_cat,
            "alpha_mean": alpha_mean,
            "alpha_var": alpha_var,
            "hard_neg_mean": hard_neg_mean,
        }
        return total_loss, metrics

    def forward(self, sample, neg_sample_list):
        loss, output, _ = self._forward_single(sample, neg_sample_list)
        return loss, output

    def predict(self, sample, neg_sample_list):
        _, pred_raw, _ = self._forward_single(sample, neg_sample_list)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -1]

        return ranking, target
