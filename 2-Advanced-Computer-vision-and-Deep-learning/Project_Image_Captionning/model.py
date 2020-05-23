import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
            Arguments
            
            - embed_size : Size of embedding.
            - hidden_size : Number of nodes in the hidden layer.
            - vocab_size : The size of vocabulary or output size.
            - num_layers : Number of layers.
        """
        super(DecoderRNN, self).__init__()
        # init embedding with the vocab_size and embed_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # init the lstm. We didn't use here the dropout parameter
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # network output
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        """
            Forward pass.
            
            Arguments :
        
            features : ouput of CNNEncoder having shape (batch_size, embed_size).
            captions : a PyTorch tensor corresponding to the last batch of captions 
                      having shape (batch_size, caption_length) .
        """
        # deleting the last column in the captions. It corresponds to the <end> token.
        captions = captions[:, :-1]
        # getting the embedding
        embed = self.word_embeddings(captions)
        # concatenate the features and the captions
        inputs = torch.cat((features.unsqueeze(1), embed), dim=1)
        # pass the inputs through the lstm layer
        r_output, _ = self.lstm(inputs)
        # fed the output of the lstm layer into the fully connected layer and get the output
        output = self.fc(r_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        '''
            accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
            
            Arguments:
            
            - inputs: pre-processed image tensor
            - states: hidden state
            - max_len: number of iteration
        '''
        # init the predicted sentence list
        predicted_sentence = []
        
        for idx in range(max_len):
            # running lstm
            output, states = self.lstm(inputs, states)
            # fed output into the fc layer
            output = self.fc(output.squeeze(1))
            # Getting the maximum probabilities
            predicted_idx = output.max(1)[1]
            # appending the predicted sentence list with the predicted index
            predicted_sentence.append(predicted_idx.item())
            # as soon as the first <end> is encountered we return the list
            # even if the Length of the captions is less than 20 at this point.
            if (predicted_idx == 1):
                return predicted_sentence
            # Update the inputs for the next iteration
            inputs = self.word_embeddings(predicted_idx).unsqueeze(1)
        
        return predicted_sentence 
            
    
    def init_weights(self):
        ''' Initialize weights of lstm '''
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
