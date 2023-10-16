from mirok.corpus import Corpus


if __name__ == '__main__':
    '''build corpus from source code data and learn word embeddings'''
    source_codes = [] # class-level source codes
    corpus: Corpus = Corpus.init_from_sources(
        source_codes,
        n_workers=24, 
    )
    corpus.learn_word_emb()
    corpus.save("corpus.pkl")