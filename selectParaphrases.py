import smart_open
from nltk.translate.bleu_score import sentence_bleu
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

def main(numPp, numTop, useBleu): 
  for inOrOut in domains:
    for dataset in domains[inOrOut]:
      for classification in classifications:
        dataset_path = 'augmentation/datasets/' + inOrOut + '_' + classification + '/' + dataset
        selectAndSave(dataset_path, numPp, numTop, useBleu)
  print("Finished")

def get_bleu_score(original, paraphrase):
    original_list = original.split()
    paraphrase_list = paraphrase.split()
    return sentence_bleu([original], paraphrase)


def selectAndSave(dataset_path, numPp, numTop, useBleu):
    input_path = dataset_path + "-questions-pps-all"
    print("Starting to select the best {} paraphrases out of the {} paraphrase candidates for each original sentence in the file '{}'".format(numTop, numPp, input_path))
    print("This uses {} as scoring".format('BLEU' if useBleu else "Cosine Similarity."))
    # Get the output path
    bleu_path = '-bleu' if useBleu else ''
    output_path = dataset_path + '-questions-pps-selected' + bleu_path

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
                paraDict[pp] = get_similarity_score(original, pp)[0][0] if not useBleu else get_bleu_score(original, pp)

        # Remove any paraphrases with similarity score of 1.0 (bc that signifies an exact match)
        non_exact_paraList = []

        # Select those with BLEU of 0.5 > s > 1 OR similarity score of 0.9 > s > 1
        if useBleu:
            non_exact_paraList = list(filter(lambda pair: pair[1] < 1 and pair[1] > 0.5, list(paraDict.items())))
        else:
            non_exact_paraList = list(filter(lambda pair: pair[1] < 1 and pair[1] > 0.9, list(paraDict.items())))

        # Sort in decreasing order by the similarity/BLEU score
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

def test():
    original = "What type of animal steals Thumbelina away?"
    pps = ["what kind of animal was stealing ?",
    "what kind of animal steals from here ?",
    "what kind of animal steal ?",
    "what kind of animal steal ?",
    "what kind of animal is steal ?",
    "what kind of animal steal ?",
    "what kind of animal rip out ?",
    "what kind of animal steal ?",
    "what a animal steal ?",
    "what kind of animal steal ?"]

    simScores = []
    bleuScores = []
    for pp in pps:
        simScores.append(get_similarity_score(original, pp, model.wv,100)[0][0])
        bleuScores.append(get_bleu_score(original, pp))

    print(simScores, "\n", bleuScores)

    original2 = "Who helps Mona with the housework and cooking?"
    pps2 = ["who helps mona with the apartment and cooking ?",
    "who helps mona and cook ?",
    "who helps mona with the house and cook ?",
    "who helps mona with the homework and cook ?",
    "who helps mona with the kitchen and cooking ?",
    "who helps mona with your house and cook ?",
    "who helps mona with cooking ?",
    "who helps mona with the dishes and cook ?",
    "who helps mona with cars and cooking ?",
    "who helps mona with the dishes and cooking ?"]

    simScores2 = []
    bleuScores2 = []
    for pp in pps2:
        simScores2.append(get_similarity_score(original2, pp, model.wv,100)[0][0])
        bleuScores2.append(get_bleu_score(original2, pp))

    print(simScores2, "\n", bleuScores2)

    original3 = "where are the vampires sailing to ?"
    pps3 = ["where are vampires sailing ?",
    "where 's vampires sailing ?",
    "where are vampires sailing ?",
    "where are vampires sailing ?",
    "where are the vampires sailing ?",
    "where are the vampires sailing ?",
    "where 's vampires sailing ?",
    "where 's the vampires flying ?",
    "where are the vampires sail ?",
    "where are vampires going to go ?"]

    simScores3 = []
    bleuScores3 = []
    for pp in pps3:
        simScores3.append(get_similarity_score(original3, pp, model.wv,100)[0][0])
        bleuScores3.append(get_bleu_score(original3, pp))

    print(simScores3, "\n", bleuScores3)


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
    parser.add_argument(
        "--useBleu", type=bool, help=("Use BLEU scoring if true, else use cosine similarity"), default=False
    )
    parser.add_argument(
        "--test", type=bool, help=("For testing purposes"), default=False
    )
    args = parser.parse_args()

    if (args.test):
        test()
    else:
        main(args.numPP, args.numTop, args.useBleu)