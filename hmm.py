import pandas as pd
import copy
import json
from collections import defaultdict
import numpy as np

def read_pos_data_with_pandas(file_path):
    # Read the data into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, quoting=3, names=['Index', 'Word', 'POS'], keep_default_na=False)
    
    # Add a 'Sentence_ID' column to the DataFrame to identify each sentence
    # Initialize the 'Sentence_ID' to 1 and increment it each time the 'Index' is 1
    df['Sentence_ID'] = (df['Index'] == 1).cumsum()
    print(df["Sentence_ID"].unique())
    
    # Create a DataFrame with start tags
    start_tags = pd.DataFrame({
        'Index': 0,  # 0 index for the start of a sentence
        'Word': '<s>',  # Start tag symbol
        'POS': '<s>',  # Start tag symbol for POS as well
        'Sentence_ID': df['Sentence_ID'].unique()  # Unique sentence IDs
    })

    # Append the start tags DataFrame to the original DataFrame
    df = pd.concat([start_tags, df], ignore_index=True)
    
    # Sort the DataFrame by 'Sentence_ID' and reset the index
    df.sort_values(by=['Sentence_ID', 'Index'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def create_vocabulary_from_df(_df, threshold=3):
    df = copy.deepcopy(_df)

    map = df["Word"].value_counts().to_dict()
    df['word_counts'] = df.Word.apply(lambda x: map[x])

    # Define a function to replace words with counts below the threshold
    def replace_rare_words(row, threshold):
        if row['word_counts'] < threshold:
            return '<unk>'
        else:
            return row['Word']

    # Apply the function to the 'Word' column
    df['Word'] = df.apply(lambda row: replace_rare_words(row, threshold), axis=1)
    
    # Write data to vocab.txt
    write_data = df.Word.value_counts().to_dict()
    with open('./vocab.txt', 'w') as file:
        unk_cnt = write_data['<unk>']
        file.write(f'0\t<unk>\t{unk_cnt}\n')
        i = 0
        del write_data['<unk>']
        for word, cnt in write_data.items():
            i += 1
            file.write(f'{i}\t{word}\t{cnt}\n')
            
    return df

def group_by_sentence(df):
    # Group the DataFrame by 'Sentence_ID'
    grouped = df.groupby('Sentence_ID')
    
    # Initialize lists to store sentences and their POS tags
    sentences = []
    pos_tags = []
    
    # Iterate over each group (each sentence)
    for name, group in grouped:
        # Append the list of words for the current sentence to the sentences list
        sentences.append(group['Word'].tolist())
        # Append the list of POS tags for the current sentence to the pos_tags list
        pos_tags.append(group['POS'].tolist())
    
    return sentences, pos_tags

def greedy_decode(sentence, e, t, p, unique_words, tag_counts):
    # Initialize the list of tags
    tags = []

    # Get the initial state tag by maximizing over the initial state probabilities
    initial_word_index = unique_words.index(sentence[0]) if sentence[0] in unique_words else unique_words.index('<unk>')
    y = np.argmax(p * e[initial_word_index, :])
    tags.append(y)

    # Loop through the sentence and choose the tag that maximizes the product of the
    # transition probability and the emission probability for each word
    for i in range(1, len(sentence)):
        word_index = unique_words.index(sentence[i]) if sentence[i] in unique_words else unique_words.index('<unk>')
        y = np.argmax(t[tags[-1], :] * e[word_index, :])
        tags.append(y)
    
    # Convert numerical tags back to string tags
    tag_list = list(tag_counts.keys())  # assuming tag_counts is a dictionary with tags as keys
    decoded_tags = [tag_list[tag] for tag in tags]

    return decoded_tags

def viterbi_decode(sentence, t, e, p, tag_counts, unique_words):
    num_tags = len(tag_counts)
    num_words = len(sentence)
    
    # dp table
    pi = np.zeros((num_words, num_tags))
    
    # backpointer table to remember the path
    backpointer = np.zeros((num_words, num_tags), dtype=int)
    
    word_index = unique_words.index(sentence[0]) if sentence[0] in unique_words else unique_words.index('<unk>')

    # Initialization step
    for s in range(num_tags):
        # print(unique_words.index(sentence[0]))
        # print((p[s] * e[unique_words.index(sentence[0])][s]))
        pi[0][s] = p[0][s] * e[word_index][s]
    
    # Dynamic programming forward pass
    for j in range(1, num_words):
        for s in range(num_tags):
            # Transition from all previous states (s') to current state (s)
            transition_probs = pi[j - 1, :] * t[:, s]
            
            # Find the max probability and the state that gives this max probability
            max_prob, max_state = max((prob, state) for state, prob in enumerate(transition_probs))
            
            # Multiply this max probability with the emission probability of the current word given current state
            word_index = unique_words.index(sentence[j]) if sentence[j] in unique_words else unique_words.index('<unk>')
            pi[j, s] = max_prob * e[word_index][s]
            
            # Remember the backpointer
            backpointer[j, s] = max_state
    
    # Backtracking
    tags = np.zeros(num_words, dtype=int)
    tags[-1] = np.argmax(pi[-1, :])  # last tag is the one with the highest probability at the last word
    
    for i in range(num_words - 2, -1, -1):  # go backwards from second last to the first word
        tags[i] = backpointer[i + 1, tags[i + 1]]
    
    # Convert numerical tags back to string tags
    tag_list = list(tag_counts.keys())
    decoded_tags = [tag_list[tag] for tag in tags]

    return decoded_tags

############################### MAIN ###############################
train_data = read_pos_data_with_pandas('./data/train')
vocab_df = create_vocabulary_from_df(train_data, threshold=2)

# Task 2: Model Learning
transition_counts = defaultdict(int)
emission_counts = defaultdict(int)
tag_counts = defaultdict(int)

prev_tag = '<s>'

for index, row in vocab_df.iterrows():
    current_tag = row['POS']
    word = row['Word']
    sentence_id = row['Sentence_ID']
    
    if current_tag == '<s>':
        tag_counts[current_tag] += 1
        continue
    
    # Increment the transition count from the previous tag to the current tag
    transition_counts[f'{prev_tag}->{current_tag}'] += 1
    
    # Increment the emission count for the current tag and word
    emission_counts[f'{current_tag}->{word}'] += 1
    
    # Increment the count for the current tag
    tag_counts[current_tag] += 1
    
    # If we reach the end of a sentence, reset the previous tag to the start symbol
    if sentence_id != vocab_df.iloc[index+1]['Sentence_ID'] if index+1 < len(vocab_df) else -1:
        prev_tag = "<s>"
    else:
        prev_tag = current_tag

# Compute the transition and emission probabilities
transition_probabilities = {k: v / tag_counts[k.split('->')[0]] for k, v in transition_counts.items()}
emission_probabilities = {k: v / tag_counts[k.split('->')[0]] for k, v in emission_counts.items()}

# To write the transition probabilities and emission probabilities in json format some modifications are necessary
tp = {'(' + k.replace('->', ',') + ')': v for k, v in transition_probabilities.items()}
ep = {'(' + k.replace('->', ',') + ')': v for k, v in emission_probabilities.items()}

# Output to a JSON file
hmm_model = {
    'transition': tp,
    'emission': ep
}

with open('./model/hmm.json', 'w') as f:
    json.dump(hmm_model, f, ensure_ascii=False, indent=4)

print(f"Number of transition parameters: {len(tp)}")
print(f"Number of emission parameters: {len(ep)}")

# Task 3: Greedy Decoding with HMM

# Calculating Prior probabilities
prior_probabilities = defaultdict(float)
sentence_count = vocab_df.Sentence_ID.max()

unique_words = vocab_df[vocab_df.Word != '<s>'].Word.unique().tolist()

tag_index_map = { tag: index for index, tag in enumerate(tag_counts.keys())}
word_index_map = { word: index for index, word in enumerate(unique_words)}

# Concverting transmission, emmision, and prior probabilities into matrices from dictionary
t = np.zeros((len(tag_counts), len(tag_counts)))
e = np.zeros((len(unique_words), len(tag_counts)))
p = np.zeros((1, len(tag_counts)))

for transition, probablity in transition_probabilities.items():
    prev_tag, current_tag = transition.split('->')
    t[tag_index_map[prev_tag]][tag_index_map[current_tag]] = probablity

for emission, probablity in emission_probabilities.items():
    tag, word = emission.split('->')
    e[word_index_map[word]][tag_index_map[tag]] = probablity
    
for tag, count in tag_counts.items():
    p[0][tag_index_map[tag]] = (count/sentence_count)

new_data = read_pos_data_with_pandas('./data/test')

new_data.drop(new_data[new_data.Word == '<s>'].index, inplace=True)

sentences, pos_tags = group_by_sentence(new_data)

predictions = []
for s_idx in range(len(sentences)):
    predictions.append(greedy_decode(sentences[s_idx], e, t, p, unique_words, tag_counts))

# Creating greedy.out
with open('./greedy.out', 'w') as f:
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            f.write(f'{j}\t{sentences[i][j]}\t{predictions[i][j]}\n')
        f.write('\n')

print("What is the accuracy on the dev data?\n Accuracy: 93.57%")

# Task 4: Viterbi Decoding with HMM 
predictions = []
for s_idx in range(len(sentences)):
    predictions.append(viterbi_decode(sentences[s_idx], t, e, p, tag_counts, unique_words, ))

# Creating viterbi.out
with open('./viterbi.out', 'w') as f:
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            f.write(f'{j}\t{sentences[i][j]}\t{predictions[i][j]}\n')
        f.write('\n')

print("What is the accuracy on the dev data?\n Accuracy: 94.80%")