import json
import os
import collections
import uuid

domains = {
#   'indomain': ['nat_questions', 'newsqa', 'squad'],
  'oodomain': ['duorc', 'race', 'relation_extraction']
}

classifications = ['train', 'val']

def construct(originalToParaphrases, dataset_path):
    # Make the containing directory and the output file that will contain the augmented dataset
    output_path = dataset_path + "-augmented"
    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except:
            print ("Failed making directory for the augmented dataset files")
    output_file = open(output_path, 'w')

    # Construct the new dataset dictionary #LEFTOFF HERE
    newDataset = dict()
    input_file = open(dataset_path, "r")

    text = input_file.read()
    text_dict = json.loads(text)
    # Make empty 'data' dict in the newDataset dictionary
    newDataset['data'] = []

    for dataDict in text_dict['data']:
        # Within the newDataset list, add a data dict that contains
        # 1) the title, 2) an empty 'paragraphs' list
        newDataset['data'].append({
            'title': dataDict['title'],
            'paragraphs': []
        })
        paragraphs = dataDict['paragraphs']
        for paragraph in paragraphs:
            paragraphDict = {
                'context': paragraph['context'],
                'qas': []
            }
            newDataset['data'][-1]['paragraphs'].append(paragraphDict)
            qas = paragraph['qas']
            for qa in qas:
                # Add the original question
                qaDict = {
                    'question': qa['question'],
                    'answers': qa['answers'],
                    'id': qa['id']
                }
                newDataset['data'][-1]['paragraphs'][-1]['qas'].append(qaDict)
                
                # Find the original question in the 'originalToParaphrases'
                # Add each of the paraphrases to the new dataset
                key = qa['question'] + '\n'
                if (key in originalToParaphrases):
                    paraphrases = originalToParaphrases[key]
                    for pp in paraphrases:
                        pp_no_null_terminator = pp[0:len(pp)-1]
                        paraphrase_qaDict = {
                            'question': pp_no_null_terminator,
                            'answers': qa['answers'],
                            'id': uuid.uuid4().hex
                        }
                        newDataset['data'][-1]['paragraphs'][-1]['qas'].append(paraphrase_qaDict)

    # Set the version in the newDataset dictionary
    newDataset['version'] = text_dict['version']
    json.dump(newDataset, output_file)
    output_file.close()
    input_file.close()


def getOriginalToParaDict(aug_dataset_path, saveToFile=False):
    # Open the input_file
    input_path = aug_dataset_path +  "-questions-pps-selected"
    input_file = open(input_path, 'r')
    lines = input_file.readlines()
    lineIndex = 0

    # Convert input_file content into a dictionary mapping original questions 
    # to a list of paraphrases
    originalToParaphrases = collections.defaultdict(list)
    print("Beginning converting the file '{}' into a dictionary that maps from the original questions to a list of paraphrases".format(input_path))

    while (lineIndex < len(lines)):
        original = lines[lineIndex]
        numPp = int(lines[lineIndex + 1])
        for cnt in range(numPp):
            paraphrase = lines[lineIndex + (cnt + 1) * 2]
            originalToParaphrases[original].append(paraphrase)
        lineIndex += (numPp + 1) * 2
    input_file.close()

    # If saveToFile, store the dictionary from original questions 
    # to its paraphrases in the output_file
    if (saveToFile):
        # Make the containing directory and the output file
        output_path = aug_dataset_path +  "-pps-selected-json"
        if not os.path.exists(os.path.dirname(output_path)):
            try:
                os.makedirs(os.path.dirname(output_path))
            except:
                print ("Failed making directory for the -pps-selected-json files")
        output_file = open(output_path, 'w')
        json.dump(originalToParaphrases, output_file, indent = 4)
        output_file.close()
    return originalToParaphrases


def main():
    for inOrOut in domains:
        for dataset in domains[inOrOut]:
            for classification in classifications:
                aug_dataset_path = 'augmentation/datasets/' + inOrOut + '_' + classification + '/' + dataset
                dataset_path = 'datasets/' + inOrOut + '_' + classification + '/' + dataset
                originalToParaphrases = getOriginalToParaDict(aug_dataset_path, False)
                construct(originalToParaphrases, dataset_path)
    print("Finished")




if __name__ == "__main__":
  print("This script is used to add the paraphrase questions to dataset text files.")
  main()