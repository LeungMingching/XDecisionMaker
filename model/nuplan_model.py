import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class NuPlanModel(BaseModel):
	def __init__(self, 
        dim_feature = 8,
	    dim_pred = 3,
	    dim_feedforward = 32,
	    n_head = 1,
	    dropout = 0.1,
	    n_encoder_layers = 1
	):
		super().__init__()
		
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=dim_feature, dim_feedforward=dim_feedforward, nhead=n_head, dropout=dropout
		)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoder_layers)
		self.pred_layer = nn.Sequential(
			nn.Linear(dim_feature, dim_feature),
			nn.ReLU(),
			nn.Linear(dim_feature, dim_pred),
		)

	def forward(self, observation):
		"""
		args:
			observation: (batch size, length, dim_feature)
		return:
			out: (batch size, n_spks)
		"""
		# out: (length, batch size, d_model)
		out = observation.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder(out)
		# out: (batch size, length, d_model)
		out = out.transpose(0, 1)
		# mean pooling
		stats = out.mean(dim=1)
		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out