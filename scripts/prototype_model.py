import torch.nn.functional as F
#from sentence_transformers import util
from torch import nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

import transformers

from transformers import Trainer, TrainerCallback

import torch

from tqdm import tqdm

import torch.nn.functional as F
from torch import nn

from scripts.utils import KMeans_cosine


class PrototypeModel(transformers.PreTrainedModel):

    def __init__(self, roberta, config, num_prototypes_per_class, definitions=None, N_input=1, N_output=100, device='cuda'):
        super(PrototypeModel, self).__init__(config)
        
        self.roberta = roberta
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        
        self.device = device

        self.num_labels = N_output
        self.clst = 0.10
        self.sep = 0.05
        
        # prototype layers
        self.epsilon = 1e-4
        self.prototype_shape = (N_output * num_prototypes_per_class, 768)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        self.prototype_sample = nn.Parameter(torch.ones(N_output, num_prototypes_per_class, 512))
        self.definition_tokens = definitions

        with torch.no_grad():
            self.definitions = (nn.Parameter(self.roberta(definitions).last_hidden_state[:,0,:]) if definitions is not None else None)
        
        self.num_prototypes = self.prototype_shape[0]
        
        layer_dim = (self.num_prototypes+N_output if definitions is not None else self.num_prototypes)
        self.last_layer = nn.Linear(layer_dim, N_output,
                                    bias=False)  # do not use bias
        assert (self.num_prototypes % N_output == 0)
        
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    N_output)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        if self.definitions is not None:
            positive_one_weights_locations = torch.cat((torch.t(self.prototype_class_identity), torch.eye(self.num_labels)), dim=1)
        else:
            positive_one_weights_locations = torch.t(self.prototype_class_identity)

        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

        
    def prototype_distances(self, x, prototype_vectors): # taking cosine similarity
        
        xp = torch.mm(x, torch.t(prototype_vectors))
        
        #print(xp, x)
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(prototype_vectors ** 2, dim=1, keepdim=True))
        
        distance = torch.nn.functional.normalize(distance, dim=1, p=2)
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        
        
        #print(distance, similarity)
        return similarity, distance


    def forward(self, input_ids, attention_mask, labels=None):
        #print(input_ids.shape)
        X = self.roberta(input_ids = input_ids, attention_mask = attention_mask)
        #print(X.hidden_states[-1].shape)
        embed = X.last_hidden_state[:,0] # encoding of CLS token as a baseline for computation cost
        
        #print(x.shape, embed.shape)
        prototype_activations, min_distances = self.prototype_distances(embed, self.prototype_vectors)
        if self.definitions is not None:
            def_activations, def_min_distances = self.prototype_distances(embed, self.definitions.to(self.device))
            logits = self.last_layer(torch.cat((prototype_activations.view(prototype_activations.size(0), -1), def_activations.view(def_activations.size(0), -1)), dim=1))
        else:
            def_activations = None
            def_min_distances = None

            logits = self.last_layer(prototype_activations.view(prototype_activations.size(0), -1))
        #print(prototype_activations.shape, def_activations.shape)
        #print(logits.shape)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return SequenceClassifierOutput(logits=(embed, logits, def_activations), loss=loss, hidden_states=(X.last_hidden_state, min_distances, def_min_distances))
    
class PrototypeCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, model, train_dataloader, **kwargs):
        #print(model.num_labels, model.device)
        device = model.device
        #if state.epoch >= 6 and state.epoch % 5 == 0:
        if state.epoch % 5 == 0:
            model.eval()
            x_per_class = [[] for i in range(model.num_labels)]
            samples = [[] for i in range(model.num_labels)]
            p_per_class = int(model.num_prototypes/model.num_labels)
            with torch.no_grad():
                for data in tqdm(train_dataloader, desc="obtaining hidden states"):
                    # encoding data to hidden state - dim should be (N, D) - N batch size, D hidden vector dim 
                    labels = data.pop("labels")
                    outputs = model(input_ids=data['input_ids'].to(device), attention_mask=data['attention_mask'].to(device))
                    logits = outputs.logits[0].cpu()
                    #print(logits.shape)
                    # reversing one-hot encoding of the labels
                    labs = (labels == 1).nonzero()

                    #print(labs)
                    labels = {}
                    for l in labs:
                        if l[0].item() in labels:
                            labels[l[0].item()].append(l[1].item())
                        else:
                            labels[l[0].item()] = [l[1].item()]
                    #print(labels)
                    # adding each hidden state to its respective class' list
                    for k, v in labels.items():
                        for c in v:
                            #print(c)
                            x_per_class[c].append(logits[k, :])
                            samples[c].append(data['input_ids'][k,:])
                print('Updating prototypes with this many samples per class')
                print([len(x) for x in x_per_class])
                # projecting prototypes for each class
                for i in tqdm(range(model.num_labels), desc='updating prototypes'):
                    # obtain cluster centers through k-means with cosine similarity
                    cl, c = KMeans_cosine(torch.stack(x_per_class[i], dim=0), K=p_per_class)

                    #print(cl, c)
                    # initial prototypes - if there's no samples closest to a certain cluster, we keep these as 
                    prototypes = [(x, -1) for x in c] + [(torch.randn(logits.size(1)), -1) for i in range(abs(c.size(0) - p_per_class))]
                    
                    for j in range(len(cl[0])):
                        p_n = cl[1][j] # the prototype that this point is closest to
                        s_n = cl[0][j] # similarity of this point to prototype
                        #print(p_n, s_n)
                        if s_n > prototypes[p_n.item()][1]:
                            prototypes[p_n.item()] = (x_per_class[i][j], s_n.item())
                            model.prototype_sample.data[i][p_n.item()] = samples[i][j]
                    #print((model.prototype_class_identity[:, i].long() == 1).nonzero(), len(prototypes))
                    model.prototype_vectors.data[(model.prototype_class_identity[:, i].long() == 1).nonzero()] = torch.stack([x[0] for x in prototypes], dim=0).unsqueeze(1).to(model.device)
                
                if model.definition_tokens is not None:
                    model.definitions.data = model.roberta(model.definition_tokens).last_hidden_state[:,0,:].to(model.device)
                
                
                    print(model.prototype_sample.data.shape)
                    for i in range(model.prototype_sample.data.size(0)):
                        for j in range(model.prototype_sample.data.size(1)):
                            with open(f'prototypes-{model.num_labels}_{i}_{j}.txt', 'w') as f:
                                text = self.tokenizer.decode(model.prototype_sample.data[i,j,:].int())
                                print(text)
                                f.write(text)
                

class PrototypeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        
        outputs = model(**inputs)
        logits = outputs.logits[1]
        
        #print(labels.shape, logits.shape)
        min_distances = outputs.hidden_states[1]
        
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.num_labels), 
                        labels.float().view(-1, self.model.num_labels))
        #cluster loss
        
        labs = (labels == 1).nonzero()
        
        prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, [x[1] for x in labs]]).to(self.model.device)
        query = torch.zeros(labels.size(0), prototypes_of_correct_class.size(1)).to(self.model.device)

        for i in range(labels.size(0)): # for each sample in batch
            for j in range(len(prototypes_of_correct_class)): # for each prototype of the gold label
                if labs[j][0] == i: # for each gold label, if the prototype is correct
                    query[i] = query[i] + prototypes_of_correct_class[j]
        
        prototypes_of_correct_class = query
        
        cluster_cost = torch.mean(torch.min(min_distances * prototypes_of_correct_class, dim=1)[0])

        #seperation loss
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        separation_cost = -torch.mean(torch.min(min_distances * prototypes_of_wrong_class, dim=1)[0])

        #sparsity loss
        l1_mask = 1 - torch.t(self.model.prototype_class_identity).to(self.model.device)
        l1 = (self.model.last_layer.weight * l1_mask).norm(p=1)

        #diversity loss
        ld = 0
        for k in range(self.model.num_labels):
            p = self.model.prototype_vectors[k*self.model.num_prototypes: (k+1)*self.model.num_prototypes]
            p = F.normalize(p, p=2, dim=1)
            matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(self.model.device) - 0.3
            matrix2 = torch.zeros(matrix1.shape).to(self.model.device)
            ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))
        
        if model.definition_tokens is not None:
            # definition loss
            def_similarity = torch.t(outputs.logits[2])
            match = torch.mm(labels, def_similarity) # version 1: only taking cosine similarity of the gold label citations
            l_def = torch.mean(match)
        else:
            l_def = 0
        
        loss = loss + self.model.clst*cluster_cost + self.model.sep*separation_cost + 0.05 * l1 + 0.00 * ld + self.model.clst*l_def #add term for cosine similarity to definition
        
        
        
        return (loss, outputs) if return_outputs else loss
        
        