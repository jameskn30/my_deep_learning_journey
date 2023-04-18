import rouge

def get_oracle_ids(sentences, highlights, summary_len = -1, include_rouge_l = False, rouge_threshold = 0.6):
    '''
    @args: 
        sentences: list, ['sent1 ... ', 'sent2 ... ', ...]
        highlights: list, ['highlight1 ...', 'highlight2 ...', ...]
    @return:
        oracle_ids: list(int), index in sentences to be used as oracle summary
    '''
    rouge_cal = rouge.Rouge()
    oracle_ids = []

    def _get_rouge_total_score(score, include_rouge_l = include_rouge_l):
        return score[0]['rouge-1']['f'] + score[0]['rouge-2']['f'] + (score[0]['rouge-l']['f'] if include_rouge_l else 0)

    for i in range(len(sentences)):
        for j in range(len(highlights)):
            score = rouge_cal.get_scores(sentences[i], highlights[j])
            total_score = _get_rouge_total_score(score, include_rouge_l)
            oracle_ids.append((i, j, total_score))
    
    if summary_len == -1: 
        oracle_ids = filter(lambda x: x[2] >= rouge_threshold, oracle_ids)
    else:
        oracle_ids = sorted(oracle_ids, key = lambda x: x[2], reverse=True)
        oracle_ids = oracle_ids[:summary_len]
    
    oracle_ids = map(lambda x: x[0], oracle_ids)
    oracle_ids = list(oracle_ids)
    
    return sorted(oracle_ids)