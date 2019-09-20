import csv
from torchtext import data
from torchtext.data import TabularDataset

file_name = '../../data/web_scraped_data.csv'
dataset_path = '../../data/skill_list.txt'

train_data = []
dev_data = []

train_data.append([])
train_data.append([])

dev_data.append([])
dev_data.append([])

train_len = 15000
i = 0

with open(file_name) as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames

    for row in reader:
        i += 1
        curr_description = row['description'].lower()
        curr_skills = []

        for field in fields:
            if field == 'description':
                continue
            else:
                curr_skills.append(int(row[field]))
                
        if i <= train_len:
            train_data[0].append(curr_description)
            train_data[1].append(curr_skills)
        else:
            dev_data[0].append(curr_description)
            dev_data[1].append(curr_skills)


dataloaders_dict = {}
dataloaders_dict['train'] = train_data
dataloaders_dict['val'] = dev_data

def get_datafields(TEXT, LABEL):
    datafields = []
    datafields.append(('description', TEXT ))
    
    f = open(dataset_path, 'r')
    for line in f.readlines():
        skill = line.replace('\n','')
        datafields.append((skill, LABEL))
    f.close()
    return datafields

def get_skill_list():
      ''' gets the complete list of skills from the text file  '''
      skill_list = []
      f = open(dataset_path, 'r')
      for line in f.readlines():
            skill = line.replace('\n','')
            skill_list.append(skill)
      f.close()
      return skill_list

        
def load_dataset(batch_size,test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
#    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    LABEL = data.LabelField()
    datafields = get_datafields(TEXT, LABEL)
    train_data = TabularDataset( path = dataset_path, format = 'csv', skip_header = True, fields = datafields   )

    train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=batch_size, sort_key=lambda x: len(x.description), repeat=False, shuffle=False)
    # train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)


    return train_iter, valid_iter

#    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter

class BatchWrapper:
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x 

      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper

                  if self.y_vars is not None: # we will concatenate y into a single tensor
                        y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
                  else:
                        y = torch.zeros((1))

                  yield (x, y)

      def __len__(self):
            return len(self.dl)

           
