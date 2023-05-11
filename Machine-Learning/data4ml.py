import jieba 
from collections import Counter

def read_dataset(path):
    labels = []
    inputs = []
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            sample = line.split('\t')
            inputs.append(sample[0])
            labels.append(sample[1])
    return inputs, labels


class MyDataset():
    def __init__(self) -> None:
        self.vocab = {}
        self.stop_words = []
    
    def set_stopword(self, path='data/scu_stopwords.txt'):
        with open(path, 'r', encoding='utf-8') as fr:
            self.stop_words = [line.strip() for line in fr.readline()]

    def build_vocab(self, inputs, max_size=5000, min_freg=5):
        cnt = {} #临时词典存储词频
        for data in inputs:
            data = jieba.lcut(data)
            for word in data:
                if word not in cnt:
                    cnt[word] = 1
                else:
                    cnt[word] += 1
        cnt = sorted([_ for _ in cnt.items() if _[1]>=min_freg and _[0] not in self.stop_words], key=lambda t:t[1], reverse=True )
        self.vocab['<pad>'] = 0

        if len(cnt) > max_size:
            for w, _ in cnt:
                if(len(self.vocab) == max_size):
                    break
                self.vocab[w] = len(self.vocab)
        else:
            for w, _ in cnt:
                self.vocab[w] = len(self.vocab)
    def transform(self, inputs, flag = 0):
        '''
        文本转换为向量
        '''
        samples = []
        for doc in inputs:
            if flag == 0:
                doc = jieba.cut(doc)
                wordset = set(doc)
                sample = []
                for word in self.vocab.keys():
                    if word in wordset:
                        sample.append(1)
                    else:
                        sample.append(0)
            elif flag == 1:
                sample = [0 for i in range(len(self.vocab))]
                word_count = Counter(doc)
                for word in word_count.items():
                    if word[0] in self.vocab.keys():
                        id = self.vocab[word[0]]
                        sample[id] = word[1]
            samples.append(sample)
        return samples
    
if __name__ == '__main__':
    inputs, labels = read_dataset('data/train.txt')
    ds = MyDataset()
    MAX_VOCAB_SIZE = 100
    ds.build_vocab(inputs, MAX_VOCAB_SIZE)
    samples = ds.transform(inputs[:5])
    print(samples[0])