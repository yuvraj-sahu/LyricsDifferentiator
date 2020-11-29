"""
Lyrics Differentiator

I have repeatedly used ts and tb as abbreviations for Taylor Swift and The
Beatles, respectively. 0 is used to represent Taylor Swift, and 1 is used to
represent The Beatles. Also, I have commented out the code that uses the SVM
classifier, since it took a very long time to train and evaluate.
"""

from nltk.stem.porter import PorterStemmer
from random import shuffle
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def filter_line(line):
    """Filters and returns the line."""
    
    #Strips the line of surrounding whitespace and stores it in a new variable
    filtered_line = line.strip()
    
    #If the line begins with '[' or '{', then this is not included in our data 
    #(since these are likely just annotations and not actual lyrics)
    if filtered_line[0] in ['[', '{']:
        return ''
    
    #Removes punctuation
    punctuation_list = ['.', ',', '[', ']', '(', ')', '{', '}', '!', '?', '\'',
                        '\"', ';', ':', '-']
    for char in punctuation_list:
        filtered_line = filtered_line.replace(char, '')
    
    #Makes the line lowercase
    filtered_line = filtered_line.lower()
    
    #Splits the line by space
    word_list = filtered_line.split(' ')
    
    #Replaces each word with its stem and removes blank words            
    stemmed_word_list = []
    stemmer = PorterStemmer()
    for word in word_list:
        if word.strip() != '':
            stemmed_word_list.append(stemmer.stem(word))
        
    #Rejoins the word list
    filtered_line = ' '.join(stemmed_word_list)
    
    #Returns this filtered line
    return filtered_line
    
def filter_data(string):
    """
    Filters each line of the lyrics and adds it to the appropriate array. 
    Returns ts_lyrics followed by tb_lyrics.
    """
    
    #Defines arrays to hold the lyrics (each line is an individual element)
    ts_lyrics = []
    tb_lyrics = []
    #The lines (in an array, split by newline)
    lines = string.split('\n')
    
    #Goes through each line
    for line in lines:
        #Stores the artist's/band's name and the lyrics. We know that there
        #will only be these two elements when each line is split by \t
        name, lyrics = line.split('\t')
        #This is the array that we will be adding to
        adding_array = ts_lyrics if name == 'taylor_swift' else tb_lyrics
        #Filters the line and adds it to the array if it is not blank
        filtered_line = filter_line(lyrics)
        if filtered_line != '':
            adding_array.append(filtered_line)
            
    #Returns the arrays
    return ts_lyrics, tb_lyrics
        
def make_segments(array, num_segments):
    """
    Splits the array into segments that are approximately equal in length 
    (within one element of each other, since the arrays might not divide 
    evenly).
    """
    
    #Stores the array length (for calculations)
    array_length = len(array)
    #Stores the length of the smaller segments and the larger segments
    small_segment_size = array_length // num_segments
    large_segment_size = small_segment_size + 1
    #Stores the number of segments that will be "larger" (have one more term)
    num_larger_segments = array_length % num_segments
    #Stores the segments (output)
    segments = [None] * num_segments
    #Shuffles the array and stores this in a new array
    shuffled_array = array.copy()
    shuffle(shuffled_array)
    
    #To be used in the loop
    num_elements_used = 0
    #Generates the larger segments
    for i in range(num_larger_segments):
        #Stores the end index for the elements that will go into the segment.
        #We do not need a variable for the start element index because this is
        #equivalent to num_elements_used
        end_element_index = num_elements_used + large_segment_size
        #Adds the segment to the segments array
        segments[i] = shuffled_array[num_elements_used:end_element_index]
        #Updates the number of elements used
        num_elements_used = end_element_index
        
    #Generates the smaller segments
    for i in range(num_larger_segments, num_segments):
        #Stores the end index for the elements that will go into the segment.
        #We do not need a variable for the start element index because this is
        #equivalent to num_elements_used
        end_element_index = num_elements_used + small_segment_size
        #Adds the segment to the segments array
        segments[i] = shuffled_array[num_elements_used:end_element_index]
        #Updates the number of elements used
        num_elements_used = end_element_index
        
    #Returns the segments array
    return segments

def spread_nested_lists(input_list):
    """
    Given a list of lists, this function spreads the contents of each list. For
    example, the call spread_nested_lists([[1, 2], [3, 4]]) would return
    [1, 2, 3, 4].
    """
    
    #The list to be returned
    new_list = []
    #Goes through each inner list and spreads it into the new list
    for inner_list in input_list:
        new_list += inner_list
    #Returns this new list
    return new_list

def split_train_dev_data(segments, dev_data_index):
    """
    Given the index of the development data segment and the segments 
    themselves, it returns the training data followed by the development data.
    """
    
    #Stores the training segments, which will be combined
    training_segments = segments[:dev_data_index] + segments[dev_data_index+1:]
    #This will store the combined training data (so that it is not a list of 
    #lists and is instead a list of strings)
    training_data = spread_nested_lists(training_segments)    
    #Returns the training data followed by the development data. The 
    #development data is just the segment with an index of dev_data_index
    return training_data, segments[dev_data_index]

def get_vectors_and_vectorizer(ts_training_data, tb_training_data, ngram_max):
    """
    Returns the input vectors, the output vector, and the vectorizer object (in
    that order) based on the training data that was inputted
    """
    
    #Defines a vectorizer
    vectorizer = CountVectorizer(ngram_range=(1, ngram_max))
    
    #Stores the length of the tb and ts training data
    len_ts_data = len(ts_training_data)
    len_tb_data = len(tb_training_data)
    #Combines the training data
    all_training_data = ts_training_data + tb_training_data
    
    #Uses the CountVectorizer to create input vectors
    input_vectors = vectorizer.fit_transform(all_training_data).toarray()
    #Creates an output vector, where 0 represents Taylor Swift and 1 represents
    #The Beatles
    output_vector = numpy.array([0]*len_ts_data + [1]*len_tb_data)
    #Returns the input vectors and the output vector
    return (input_vectors, output_vector, vectorizer)

def test_lines(vectorizer, test_data, test_answer, predict_function):
    """
    Returns the number of correct predictions for a specific inputted 
    artist/band.
    """
    
    #Stores the test vectors
    test_vectors = vectorizer.transform(test_data).toarray()
    #Gets the results for the predictions
    results = predict_function(test_vectors)
    #Returns the number of results that were correct
    return len(numpy.where(results == test_answer)[0])
    
def test_model(fit_function, predict_function, 
               vectors_and_vectorizer, ts_test_data, tb_test_data):
    
    input_vectors, output_vector, vectorizer = vectors_and_vectorizer
    fit_function(input_vectors, output_vector)
    
    num_correct_ts = test_lines(vectorizer, ts_test_data, 0, predict_function)
    num_correct_tb = test_lines(vectorizer, tb_test_data, 1, predict_function)
    
    return num_correct_ts + num_correct_tb

def test_nb(vectors_and_vectorizer, ts_test_data, tb_test_data):
    """
    Trains and tests a Naive Bayes model and returns the number of correct 
    predictions made by it.
    """
    
    #Creates the NB model object
    nb_model = GaussianNB()
    #Returns the total number of correct predictions
    return test_model(nb_model.fit, nb_model.predict, vectors_and_vectorizer, 
                      ts_test_data, tb_test_data)
    

def test_svm(vectors_and_vectorizer, ts_test_data, tb_test_data):
    """
    Trains and tests a Support Vector Machine and returns the number of correct
    predictions made by it.
    """
    
    #Creates the SVM object
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    #Returns the total number of correct predictions
    return test_model(svm.fit, svm.predict, vectors_and_vectorizer, 
                      ts_test_data, tb_test_data)
    

#Reads from the file and gets the data
file_name = 'lyrics.txt'
file = open(file_name, mode='r')
ts_data, tb_data = filter_data(file.read())
file.close()

#In this case, we will be making 10 segments to achieve our 80%-10%-10% split
num_segments = 10

#Splits the data into 10 segments, where the first segment represents the
#testing data and the remaining segments are the training and development data;
#this will give us a 80%-10%-10% split
ts_segments = make_segments(ts_data, num_segments)
ts_train_dev, ts_test = ts_segments[1:], ts_segments[0]
tb_segments = make_segments(tb_data, num_segments)
tb_train_dev, tb_test = tb_segments[1:], tb_segments[0]

#Stores the total amount of correct guesses for each model
total_correct_nb = 0
#total_correct_svm = 0

#Implements 9-fold cross validation with the remaining 9 segments in the 
#training and development data by going through each segment index and choosing
#that segment to be the development data
for dev_data_index in range(num_segments-1):
    #Splits the training and development data based on dev_data_index
    ts_train, ts_dev = split_train_dev_data(ts_segments, dev_data_index)
    tb_train, tb_dev = split_train_dev_data(tb_segments, dev_data_index)
    
    #Gets the vectors and vectorizer
    #Using an ngram maximum of 2 yields much better results for this model
    nb_vectors_and_vectorizer = get_vectors_and_vectorizer(ts_train, 
                                                           tb_train, 2)
    #Removing duplicates significantly shortens the runtime to train the model;
    #however, it still takes a while to run
    #svm_vectors_and_vectorizer = get_vectors_and_vectorizer(
    #        list(set(ts_train)), list(set(tb_train)), 1)
    
    #Stores the number of correct predictions by each model
    num_correct_nb = test_nb(nb_vectors_and_vectorizer, ts_dev, tb_dev)
    #num_correct_svm = test_svm(nb_vectors_and_vectorizer, ts_dev, tb_dev)
    
    #Adds the number of correct guesses to the totals
    total_correct_nb += num_correct_nb
    #total_correct_svm += num_correct_svm
    
    print('Test case', dev_data_index+1, 'results:')
    print('NB correct:', num_correct_nb, 'out of', len(ts_dev) + len(tb_dev))
    #print('SVM correct:', num_correct_svm, 'out of', 
    #      len(ts_dev) + len(tb_dev))
    print()
    
#Spreads the lists and stores these new lists in new variables
ts_train_dev_spread = spread_nested_lists(ts_train_dev)
tb_train_dev_spread = spread_nested_lists(tb_train_dev)

#Prints the total number of correct guesses for each model
len_train_dev_data = len(ts_train_dev_spread) + len(tb_train_dev_spread)
print('Total correct for NB on training/development sets:', total_correct_nb, 
      'out of', len_train_dev_data)
print(f'Accuracy: ~{round(total_correct_nb / len_train_dev_data * 100, 2)}%')
#print('Total correct for SVM:', total_correct_svm, 'out of', 
#      num_train_dev_values)
#print(f'Accuracy: ~{round(total_correct_svm / len_train_dev_data * 100, 2)}%')
print()

#Trains the NB model on the training and development data set
nb_vectors_and_vectorizer = get_vectors_and_vectorizer(ts_train_dev_spread, 
                                                       tb_train_dev_spread, 2)
num_correct_nb = test_nb(nb_vectors_and_vectorizer, ts_test, tb_test)
len_test_data = len(ts_test) + len(tb_test)
print('Total correct for NB model on testing set: ', num_correct_nb, 'out of', 
      len_test_data)
print(f'Accuracy: ~{round(num_correct_nb / len_test_data * 100, 2)}%')
