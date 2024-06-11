def get_word_indicies(text):
    """Get word indicies for our text."""
    word_indicies = {}
    word_start_ind = 0
    
    for n, word in enumerate(text.split(' ')):
        word_indicies[word_start_ind] = n
        word_start_ind += len(word) + 1
    return word_indicies


def process_prediction(prediction):
    """Prediction format is for symbols, not for words
    We need to fix it
    """
    fixed_prediction = []
    
    for ent_gr in prediction:
        add_to_start = 0
        if len(ent_gr['word'].split(' ')) != 1:
            for word in ent_gr['word'].split(' '):
                fixed_prediction.append({'word': word, 'start': ent_gr['start'] + add_to_start, 
                                         'entity_group': ent_gr['entity_group']})
                add_to_start += len(word) + 1
        else:
            fixed_prediction.append(ent_gr)
    return fixed_prediction


def get_prediction(text, prediction):
    """Function for prediction generation""" 
    pred_labels = {}
    ent_cnt = 0
    base_labels = ['O'] * len(text.split(' '))
    word_ind = get_word_indicies(text)
    fixed_prediction = process_prediction(prediction)
    
    for ent_gr in fixed_prediction:
        if ent_gr['entity_group'] == 'discount':
            pred_labels[word_ind[ent_gr['start']]] = 'B-discount'
        elif ent_gr['entity_group'] == 'value' and ent_cnt==0:
            pred_labels[word_ind[ent_gr['start']]] = 'B-value'
            ent_cnt += 1
        else:
            pred_labels[word_ind[ent_gr['start']]] = 'I-value'
    
    lbl_keys = pred_labels.keys()
    
    return [def_lbl if n not in lbl_keys else pred_labels[n] for n, def_lbl in enumerate(base_labels)]