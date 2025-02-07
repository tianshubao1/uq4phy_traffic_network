import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from model.pytorch.dcrnn_cell import DCGRUCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

        # ----------------  pde init-----------------------------------
        self.batch_size = 128
        self.v_f = 70.86        #PEMSD8 free-speed
        self.rho_max = 0.649    #PEMSD8, rho_max,  this hyperparameter needs to be carefully chosen
        self.delta_t = 300/3600     # 5 min        
        
        PEMSD8_matrix = np.load('data/sensor_graph/PEMSD8_matrix.npy')            
        self.weight_matrix = torch.from_numpy(PEMSD8_matrix).to(device)        
        self.weight_matrix[self.weight_matrix < 0.005] = 0       
        self.source_matrix = torch.transpose(self.weight_matrix, 0, 1)
        

        self.h_input = None
        self.lmbda = nn.Parameter(torch.FloatTensor(1).to(device))      
        self.beta = nn.Parameter(torch.FloatTensor(1).to(device))          
        init.normal_(self.lmbda, mean=0.5, std=0.1)    
        init.normal_(self.beta, mean=0.5, std=0.1)   
        
        self.scale1 = nn.Parameter(torch.FloatTensor(1).to(device))  
        self.scale2 = nn.Parameter(torch.FloatTensor(1).to(device)) 
        init.normal_(self.scale1, mean=0.5, std=0.1)  
        init.normal_(self.scale2, mean=0.5, std=0.1) 
        
    def flow(self, rho, v):       # [v^2/2 - (v_f*rho/rho_max) * v]
        f = torch.square(v)/2 - torch.mul((self.v_f*rho/self.rho_max), v)     # f:(120, 44)   elementwise product, not matrix multiply
        
        return f
        
    def forward(self, inputs, batch, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)  torch.Size([128, 414]) 
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        
        #-------------add pde------------------
       
   
        #inputs: torch.Size([128, 414]) 
        #batch: torch.Size([128, 414])           
        size1 = inputs.size()[1]/ self.num_nodes  
        size1 = int(size1)  #size1: 2,  number of features
        #print(size1)
        
        x_input = inputs.view(self.batch_size, size1, self.num_nodes)       #    [128, 2, 207]       
        
        if self.h_input is None:       
            self.h_input = nn.Parameter(torch.FloatTensor(1, size1, self.num_nodes).to(device))      # self.h: (2, 207)
            init.normal_(self.h_input, mean=0.0, std=0.1)  
        
        # ----------------------------- x_input ----------------------------------
        
        batch = batch.view(self.batch_size, size1, self.num_nodes)  # batch: (128, 2, 207)
        rho_input = batch[:, 1, :]        # occupancy   rho_input:(128, 207)
        rho_input = rho_input.view(self.batch_size, 1, self.num_nodes)    #rho_input:(128, 1, 207)
        broad_input = torch.zeros(self.batch_size, size1, self.num_nodes).to(device)   # rho_input:(128, size1, 207)
        rho_input = rho_input + broad_input     #rho_input:(128, 2, 207)
        
        
        curr_flux = self.flow(rho_input, x_input)            
        flux = self.delta_t*(torch.matmul(curr_flux, self.source_matrix) - torch.matmul(curr_flux, self.weight_matrix))/2   
        x_input_next = x_input - self.scale1*flux + self.scale2*self.delta_t*self.h_input      
        
        
        x_input_next = torch.sigmoid(x_input_next)         
        x_input_next =  self.lmbda*x_input_next + self.beta*x_input  
        
        inputs = x_input_next.view(self.batch_size, size1 *self.num_nodes)          #torch.Size([128, 414])  
        #print(inputs.size())

        # ----------- add to orignial inputs -----------------      
        
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], batch)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim*3)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

       # ----------------  pde init-----------------------------------
        self.batch_size = 128
        self.v_f = 70.86
        self.rho_max = 0.649    #METR-LA, rho_max,  this hyperparameter needs to be carefully chosen
        self.delta_t = 300/3600     # 5 min          
        
        PEMSD8_matrix = np.load('data/sensor_graph/PEMSD8_matrix.npy')  
        self.weight_matrix = torch.from_numpy(PEMSD8_matrix).to(device)              
        self.weight_matrix[self.weight_matrix < 0.005] = 0       
        self.source_matrix = torch.transpose(self.weight_matrix, 0, 1)
        

        self.h_input = None
        self.lmbda = nn.Parameter(torch.FloatTensor(1).to(device))      
        self.beta = nn.Parameter(torch.FloatTensor(1).to(device))          
        init.normal_(self.lmbda, mean=0.5, std=0.1)    
        init.normal_(self.beta, mean=0.5, std=0.1)   
        
        self.scale1 = nn.Parameter(torch.FloatTensor(1).to(device))  
        self.scale2 = nn.Parameter(torch.FloatTensor(1).to(device)) 
        init.normal_(self.scale1, mean=0.5, std=0.1)  
        init.normal_(self.scale2, mean=0.5, std=0.1) 
        
    def flow(self, rho, v):       # [v^2/2 - (v_f*rho/rho_max) * v]
        f = torch.square(v)/2 - torch.mul((self.v_f*rho/self.rho_max), v)     # f:(120, 44)   elementwise product, not matrix multiply
        
        return f
        
        
    def forward(self, inputs, batch, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)     # torch.Size([128, 207])
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        
        
        #-------------add pde------------------
       
        #batch: torch.Size([128, 414])
        size1 = inputs.size()[1]/ self.num_nodes  #1
        size1 = int(size1)
        size2 = batch.size()[1]/ self.num_nodes  #2
        size2 = int(size2)
        #print(size2)
        x_input = inputs.view(self.batch_size, size1, self.num_nodes)       #    [128, 1, 207]       
        
        if self.h_input is None:       
            self.h_input = nn.Parameter(torch.FloatTensor(1, size1, self.num_nodes).to(device))      # self.h: (2, 207)
            init.normal_(self.h_input, mean=0.0, std=0.1)  
        
        # ----------------------------- x_input ----------------------------------
        
        batch = batch.view(self.batch_size, size2, self.num_nodes)  # batch: (128, 2, 207)
        rho_input = batch[:, 1, :]        # occupancy   rho_input:(128, 207)
        #print(rho_input)
        rho_input = rho_input.view(self.batch_size, 1, self.num_nodes)    #rho_input:(128, 1, 207)
        broad_input = torch.zeros(self.batch_size, size1, self.num_nodes).to(device)   # rho_input:(128, size1, 207)
        rho_input = rho_input + broad_input     #rho_input:(128, 2, 207)
        
        
        curr_flux = self.flow(rho_input, x_input)            
        flux = self.delta_t*(torch.matmul(curr_flux, self.source_matrix) - torch.matmul(curr_flux, self.weight_matrix))/2   
        x_input_next = x_input - self.scale1*flux + self.scale2*self.delta_t*self.h_input      
        
        
        x_input_next = torch.sigmoid(x_input_next)         
        x_input_next =  self.lmbda*x_input_next + self.beta*x_input  
        
        inputs = x_input_next.view(self.batch_size, size1 *self.num_nodes)        
        

        # ----------- add to orignial inputs -----------------              
        
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], batch)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim,3)

        return output, torch.stack(hidden_states)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, batch):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):     # self.encoder_model.seq_len: 12 , batch : [12, 128, 414]  207 sensors
            
            #print(batch.size())
            _, encoder_hidden_state = self.encoder_model(inputs[t], batch[t, :, :], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, batch, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, batch[t, :, :],
                                                                      decoder_hidden_state)
            decoder_input = decoder_output.T[1].T
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        

        inputs_original = inputs            # inputs : torch.Size([12, 128, 414]) 
        encoder_hidden_state = self.encoder(inputs, inputs_original)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, inputs_original, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs
