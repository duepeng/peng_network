import torch
from torch import nn
import torch_scatter

class PFNLayerV2(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 use_norm=True,
				 last_layer=False):
		super().__init__()
		
		self.last_vfe = last_layer
		self.use_norm = use_norm
		if not self.last_vfe:
			out_channels = out_channels // 2

		if self.use_norm:
			self.linear = nn.Linear(in_channels, out_channels, bias=False)
			self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
		else:
			self.linear = nn.Linear(in_channels, out_channels, bias=True)
		
		self.relu = nn.ReLU()  

	def forward(self, inputs, unq_inv):

		x = self.linear(inputs)
		x = self.norm(x) if self.use_norm else x
		x = self.relu(x)
		x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

		if self.last_vfe:
			return x_max
		else:
			x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
			return x_concatenated
		
if __name__ == '__main__':
		
		# paras
		num_filters = [8, 32]
		pfn_layers = []
		use_norm = True

		for i in range(len(num_filters) - 1):
			in_filters = num_filters[i]
			out_filters = num_filters[i + 1]
			pfn_layers.append(
				PFNLayerV2(in_filters, out_filters, use_norm, last_layer=(i >= len(num_filters) - 2))
			)
		pfn_layers = nn.ModuleList(pfn_layers)

		pass