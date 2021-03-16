import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import brown  
import argparse
import os

nltk.download('brown')
sentences = brown.sents()
vector_size = 100
model = Word2Vec(sentences, size=vector_size, window=5, min_count=1, workers=4)

domains = {
#   'indomain': ['nat_questions', 'newsqa', 'squad'],
  'oodomain': ['duorc', 'race', 'relation_extraction']
}

classifications = ['train', 'val']


# Calculate the similarity score
def get_similarity_score(original, paraphrase, vectors=model.wv, vector_size=vector_size):
    original_feature = np.zeros((vector_size,))
    paraphrase_feature = np.zeros((vector_size,))
    original_list = original.split()
    paraphrase_list = paraphrase.split()
    original_len = 0
    paraphrase_len = 0
    for i in range(len(original_list)):
        if(original_list[i] in vectors):
            original_feature = np.add(original_feature, model.wv[original_list[i]])
            original_len += 1
    for i in range(len(paraphrase_list)):
        if(paraphrase_list[i] in vectors):
            paraphrase_feature = np.add(paraphrase_feature, model.wv[paraphrase_list[i]])
            paraphrase_len += 1
    if(original_len > 1):
        original_feature = np.divide(original_feature, original_len)
    if(paraphrase_len > 1):
        paraphrase_feature = np.divide(paraphrase_feature, paraphrase_len)
    return cosine_similarity(original_feature.reshape(1, -1),paraphrase_feature.reshape(1, -1))

def main(numPp, numTop): 
  for inOrOut in domains:
    for dataset in domains[inOrOut]:
      for classification in classifications:
        dataset_path = 'augmentation/datasets/' + inOrOut + '_' + classification + '/' + dataset
        selectAndSave(dataset_path, numPp, numTop)
  print("Finished")

def selectAndSave(dataset_path, numPp, numTop):
    input_path = dataset_path + "-questions-pps-all"
    print("Starting to select the best {} paraphrases out of the {} paraphrase candidates for each original sentence in the file '{}'".format(numTop, numPp, input_path))

    # Get the output path
    output_path = dataset_path + '-questions-pps-selected'

    # Make the containing directories if it doesn't already exist
    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except:
            print ("Failed extracting questions")

    # Open the output_file for writing and the input_file for reading
    output_file = open(output_path, 'w')
    input_file = open(input_path, 'r')

    groupSize = numPp + 1
    lines = input_file.readlines()
    # For each group, where a group consists of an original sentence and numPp number of paraphrases
    numGroups = int(len(lines) / groupSize)
    for groupIndex in range(numGroups):
        original = lines[groupIndex * groupSize]
        paraDict = dict()
        # Get the unique paraphrases and its similarity score
        for ppIndex in range(1, groupSize):
            lineIndex = (groupIndex * groupSize) + ppIndex
            pp = lines[lineIndex]
            if (pp not in paraDict):
                paraDict[pp] = get_similarity_score(original, pp)[0][0]

        # Remove any paraphrases with similarity score of 1.0 (bc that signifies an exact match)
        non_exact_paraList = list(filter(lambda pair: pair[1] < 1 and pair[1] > 0.9, list(paraDict.items())))
        # Sort in decreasing order by the similarity score
        non_exact_paraList.sort(key=lambda x:x[1], reverse=True)

        # Write the original to file, the number of paraphrases chosen, and each chosen paraphrases
        # (all of separate lines)
        count = min(len(non_exact_paraList), numTop)
        if (count > 0):
            output_file.write(original)
            output_file.write(str(count) + '\n')
            for candidateIndex in range(count):
                output_file.write(non_exact_paraList[candidateIndex][0] + str(non_exact_paraList[candidateIndex][1]) + '\n')

    output_file.close()
    input_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Given files with original and its paraphrases, select the best 3 and save the result."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--numPP", type=str, help=("Number of paraphrases for each original sentence."), default=10
    )
    parser.add_argument(
        "--numTop", type=str, help=("Maximum number of top paraphrases for each original sentence to pick."), default=3
    )
    args = parser.parse_args()
    main(args.numPP, args.numTop)
