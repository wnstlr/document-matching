"""
Part of the code adapted from 
https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
"""

from typing import Any, Iterable, List, Tuple, Union
from nltk.tokenize import sent_tokenize

from compute_explanation import rescale_pos_neg_attrs
import numpy as np

try:
    from IPython.core.display import HTML, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "token_input_summary",
        "attr_summary",
        "token_input_text1",
        "attr_text1",
        "token_input_text2",
        "attr_text2",
        "score1", 
        "score2", 
        "plain_input_text1",
        "plain_input_text2",
        "plain_input_summary"
    ]

    def __init__(
        self,
        token_input_summary,
        token_input_text1,
        token_input_text2,  # list of tokens for alt 
        plain_input_summary,
        plain_input_text1,
        plain_input_text2, # list of texts for alt
        attr_summary,
        attr_text1,
        attr_text2, # list of txts
        score1,
        score2 # list of scores for alts
    ) -> None:
        self.attr_summary = attr_summary
        self.attr_text1 = attr_text1
        self.attr_text2 = attr_text2
        self.score1 = score1
        self.score2 = score2
        self.token_input_summary = token_input_summary
        self.token_input_text1 = token_input_text1
        self.token_input_text2 = token_input_text2
        self.plain_input_summary = plain_input_summary
        self.plain_input_text1 = plain_input_text1
        self.plain_input_text2 = plain_input_text2

def word2token_attr(tokenizer, original_text, words, word_attr, max_length=512):
    t2w = dict()
    w2t = dict()    
    assert(len(words) == len(word_attr))
    encoded = tokenizer(original_text.replace('\n', ' '), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    # token to word mapping
    for i, t in enumerate(encoded.tokens()):
        t2w[i] = encoded.token_to_word(i)
    # word to token mapping
    for k, v in t2w.items():
        if v is None:
            if k == 0:
                w2t[-1] = k
            elif k == len(t2w)-1:
                w2t[-2] = k
        else:
            if v in w2t.keys(): 
                w2t[v].append(k)
            else:
                w2t[v] = [k]
    # convert word to token
    toks = encoded.tokens()
    toks_attr = np.zeros(len(toks))
    toks_attr[0] = 0 # special tokens
    toks_attr[-1] = 0 # special tokens
    for k, v in w2t.items():
        if k < 0: 
            continue
        for tid in v:
            toks_attr[tid] = word_attr[k]
    return toks, toks_attr, t2w, w2t

def token2word_attr(tokenizer, original_text, token_attr):
    # convert token-level attribution to word-level attribution
    encoded = tokenizer(original_text.replace('\n', ' '), truncation=True, max_length=len(token_attr))
    word_size = len(np.unique([x for x in encoded.word_ids() if x is not None]))
    word_list = [None] * word_size
    attr_mod = np.zeros(word_size)
    for i, wid in enumerate(encoded.word_ids()):
        if wid is not None:
            chars = encoded.word_to_chars(wid)
            start = chars.start
            end = chars.end
            word = original_text[start:end]
            word_list[wid] = word
            attr_mod[wid] += token_attr[i]
    assert(len(attr_mod) == len(word_list))        
    return word_list, attr_mod

def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)

def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return ''
        #return "#" + token.strip("<>")
    token.replace('\\', '')
    return token

def _get_color(attr, continuous=True, multicolor=False):
    def _match_color_scale_level(attr, max, min):
        return (max-min) * attr + min
    
    if continuous:
        if not multicolor: # COLORSCHEME for Post Hoc and Extractive
            # clip values to prevent CSS errors (Values should be from [-1,1])
            attr = max(-1, min(1, 2*attr))
            if attr > 0: # blue
                hue = 185
                sat = 75
                lig = 100 - int(40 * attr)
            else: # red
                hue = 0
                sat = 60
                lig = 100 - int(-25 * attr)
            return "hsl({}, {}%, {}%)".format(hue, sat, lig)
        else: # COLORSCHEME for highlighting both sentence and phrases
            # color the phrases
            if attr >= 10:
                sat = 100
                lig = 72
                #lig = 70
                # red
                if attr // 10 == 1: #red
                    hue = 338
                elif attr // 10 == 2: #blue
                    hue = 208
                    lig = 65
                elif attr // 10 == 3: # yellow
                    hue = 45
                    lig = 60
                else:
                    raise ValueError('invalid attr value %f'%attr)
            # color the sentences
            elif attr < 5:
                sat = 70
                if 1 <= attr <= 2: # red
                    hue = 338
                    offset = 1
                elif 2 < attr <= 3: #blue
                    hue = 208
                    offset = 2
                elif 3 < attr <= 4: # yello
                    hue = 45
                    offset = 3
                elif attr == 0: # white if attribution is zero
                    hue = 0
                    sat = 0
                    offset = attr+1
                    lig = 100
                else:
                    raise ValueError('invalid attr value %f'%attr)
                lig = 100 - _match_color_scale_level(attr-offset, 20, 10)
                #lig = 100 - ((attr-offset) * 10 + 10) # if attr is 0, lig is 90 (light color)
            else:
                raise ValueError('invalid attr value %f'%attr)
            return "hsl({}, {}%, {}%)".format(hue, sat, lig)
    else:
        # get fixed set of colors according to the attributions 
        vals = [0.1, 0.4, 0.6] # NOTE specify different discrete values each attribution can take
        if attr == vals[0]*2: # first color: magenta
            hue = 338
            sat = 100
            lig = 65
        elif attr == vals[1]*2: # second color: blue
            hue = 208
            sat = 100
            lig = 65
        elif attr == vals[2]*2: # third color:  yellow
            hue = 45
            sat = 100
            lig = 65
        elif attr == vals[0]: # first color: light magenta
            hue = 338
            sat = 70
            lig = 90
        elif attr == vals[1]: # second color: light blue
            hue = 208
            sat = 70
            lig = 90
        elif attr == vals[2]: # third color: light yellow
            hue = 45
            sat = 70
            lig = 90
        elif attr == 0: # white if attribution is zero
            hue = 0
            sat = 0
            lig = 100
        else:
            hue = 0
            sat = 0
            lig = 100
            raise ValueError('attribution value is not valid: expected %s, got %f'%(vals, attr))
            
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)
        
def remove_spaces(words, attrs):
    word_out = []
    attr_out = []
    stack = []
            
    idx = 0
    while idx < len(words):
        # stick to the word before
        if words[idx] in [',', '.', '%', ')', ').', '".', '),', '",', ',"', ",'", '."',\
                        ':', '?', ';', '?"', '"?', '),"']:
            word_out.pop()
            attr_out.pop()
            word_out.append(words[idx-1] + words[idx])
            attr_out.append(attrs[idx-1] + attrs[idx])
            idx += 1
        # stick to both words on left and right
        # elif words[idx] in ['-']:
        #     word_out.pop()
        #     attr_out.pop()
        #     word_out.append(words[idx-1] + words[idx] + words[idx+1])
        #     attr_out.append(attrs[idx-1] + attrs[idx] + attrs[idx+1])
        #     idx += 2
        # stick to the word after
        elif words[idx] in ["(", '"']:
            word_out.append(words[idx] + words[idx+1])
            attr_out.append(attrs[idx] + attrs[idx+1])
            idx += 2
        elif words[idx] == "'s":
            word_out.pop()
            attr_out.pop()
            word_out.append(words[idx-1] + words[idx])
            attr_out.append(attrs[idx-1] + attrs[idx])
            idx += 1
        else:
            word_out.append(words[idx])
            attr_out.append(attrs[idx])
            idx += 1         
    return word_out, attr_out

def format_word_importances(words, importances, left=True, summary=False, continuous=True, multicolor=False):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    #if left:
    if summary:
        tags = ['<td style="font-size:23px">']
    else:
        tags = ['<td width="33%"; style="padding-right:10px; padding-left:1px; border-left:solid; border-right:solid; vertical-align:top">']
    #else:
    #    tags = ['<td style="padding-right:1px; padding-left:10px">']
    if continuous and not multicolor:
        words_, importances_ = remove_spaces(words, importances)
    else:
        words_, importances_ = words, importances
    for word, importance in zip(words_, importances_[: len(words_)]):
        word = format_special_tokens(word)
        color = _get_color(importance, continuous=continuous, multicolor=multicolor)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black" size=5px> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)

def post_process_token_arr(arr):
    out = []
    for token in arr:
        if token == '<pad>':
            continue
        else:
            out.append(token.replace('Ġ', '').replace('Ċ', ''))
    return out

# visualize multi-level highlights
def visualize_text_multi(
    datarecords: Iterable[VisualizationDataRecord], # multi-level info
    legend: bool = True,
    show_score=True,
    show_exp=True,
    ordering=None,
) -> "HTML":  # visualize in column
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ['<table width:100% align="left">'] 
    for datarecord in datarecords:
        if show_exp and legend:
            #add legend for explanations
            dom.append(
                '<div style="border-top: 1px solid; margin-top: 5px; padding-top: 5px; display: inline-block">'
            )
            dom.append("<b>Legend: </b>")

            for value, label in zip([0.1, 0.4, 0.6], \
                ["1st Sentence", "2nd Sentence", "3rd Sentence"]):
                dom.append(
                    '<span style="display: inline-block; width: 10px; height: 10px; \
                    border: 1px solid; background-color: \
                    {value}"></span> {label}  '.format(
                        value=_get_color(value), label=label
                    )
                )
            dom.append("</div>")
            
        # Summary table
        dom.append('<tr><th><br><p style="font-size:26px">Summary</p><br></th></tr>')
        
        # show highlighted summary, in sentence level
        dom.append('<tr>{}</tr>'.format(
            format_word_importances(
                post_process_token_arr(datarecord.token_input_summary),
                datarecord.attr_summary,
                summary=True,
                continuous=True, 
                multicolor=True
            )
        ))
        
        #dom.append('<tr><td><p style="font-size:23px">{}</p><br></td></tr>'.format(datarecord.plain_input_summary))
        dom.append('</table>')
        
        # Text info 
        if not show_exp:
            attrs1, attrs2 = np.zeros(1024), [np.zeros(1024) for i in range(len(datarecord.plain_input_text2))]
        else:
            attrs1, attrs2 = datarecord.attr_text1, datarecord.attr_text2
            
        assert(type(attrs2) == list)
        
        all_scores = [datarecord.score1] + datarecord.score2
        all_texts = [datarecord.token_input_text1] + datarecord.token_input_text2 
        all_attrs = [attrs1] + attrs2

        assert(len(all_scores) == len(all_texts))
        assert(len(all_attrs) == len(all_texts))
        correct_id = 0
        if ordering is not None:
            assert(type(ordering) == list)
            assert(len(ordering) == len(all_texts))
            all_scores = [all_scores[x] for x in ordering]
            all_texts = [all_texts[x] for x in ordering]
            all_attrs = [all_attrs[x] for x in ordering]
            correct_id = ordering.index(0)
            
        rows = ['<table width:100% align="left">']
        rows.append('<tr><th colspan="2"></th></tr>')
        rows.append('<tr>')
        
        if show_score:
            for i, score in enumerate(all_scores):
                rows.append('<td><p style="font-size:26px"><b>Article %d --- <u>Score: %0.2f</u></b></p><br></td>'%(i+1,score))
            rows.append('</tr>')
            rows.append('<tr>')

        for t, attr in zip(all_texts, all_attrs):
            rows.append('{}'.format(
                format_word_importances(
                    post_process_token_arr(t), 
                    attr, 
                    left=True,
                    summary=False,
                    continuous=True, 
                    multicolor=True
                )
            ))
        rows.append('</tr></table>')

    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))
    display(html)

    return html, correct_id

def visualize_text(
    datarecords: Iterable[VisualizationDataRecord], 
    legend: bool = True,
    show_score=True,
    show_exp=True,
    ordering=None,
) -> "HTML":  # visualize in column
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ['<table width:100% align="left">']
    for datarecord in datarecords:
        if show_exp and legend:
            #add legend for explanations
            dom.append(
                '<div style="border-top: 1px solid; margin-top: 5px; padding-top: 5px; display: inline-block">'
            )
            dom.append("<b>Legend: </b>")

            for value, label in zip([-1, 0, 1], \
                ["Words that decrease the score", "No effect", "Words that increase the score"]):
                dom.append(
                    '<span style="display: inline-block; width: 10px; height: 10px; \
                    border: 1px solid; background-color: \
                    {value}"></span> {label}  '.format(
                        value=_get_color(value), label=label
                    )
                )
            dom.append("</div>")
            
        # Summary table
        dom.append('<tr><th><br><p style="font-size:26px">Summary</p><br></th></tr>')
        dom.append('<tr><td><p style="font-size:23px">{}</p><br></td></tr>'.format(datarecord.plain_input_summary))
        dom.append('</table>')
        
        # Text info 
        if not show_exp:
            attrs1, attrs2 = np.zeros(1024), [np.zeros(1024) for i in range(len(datarecord.plain_input_text2))]
        else:
            attrs1, attrs2 = datarecord.attr_text1, datarecord.attr_text2
            
        assert(type(attrs2) == list)
        
        all_scores = [datarecord.score1] + datarecord.score2
        all_texts = [datarecord.token_input_text1] + datarecord.token_input_text2 
        all_attrs = [attrs1] + attrs2

        assert(len(all_scores) == len(all_texts))
        assert(len(all_attrs) == len(all_texts))
        correct_id = 0
        if ordering is not None:
            assert(type(ordering) == list)
            assert(len(ordering) == len(all_texts))
            all_scores = [all_scores[x] for x in ordering]
            all_texts = [all_texts[x] for x in ordering]
            all_attrs = [all_attrs[x] for x in ordering]
            correct_id = ordering.index(0)
            
        rows = ['<table width:100% align="left">']
        rows.append('<tr><th colspan="2"></th></tr>')
        rows.append('<tr>')
        
        if show_score:
            for i, score in enumerate(all_scores):
                rows.append('<td><p style="font-size:26px"><b>Article %d --- <u>Score: %0.2f</u></b></p><br></td>'%(i+1,score))
            rows.append('</tr>')
            rows.append('<tr>')

        #rows.append('{}'.format(format_word_importances(
        #    post_process_token_arr(datarecord.token_input_text1), attrs1, True)))
        for t, attr in zip(all_texts, all_attrs):
            rows.append('{}'.format(format_word_importances(
                post_process_token_arr(t), attr, left=True)))
        rows.append('</tr></table>')

    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))
    display(html)

    return html, correct_id

def visualize(vis, exp_type, **kwargs):
    if exp_type == 'phrase' or exp_type == 'rouge_phrase':
        out, correct_answer = visualize_text_multi([vis], **kwargs)
    else:
        out, correct_answer = visualize_text([vis], **kwargs)
    return out, correct_answer

# storing couple samples in an array for visualization purposes
def generate_question(text_info, 
                      text_idx, 
                      show_exp=False,
                      exp_type='heuristic',
                      show_score=False, 
                      legend=True,
                      shuffle=True,
                      none_of_above=False):
    # set the correct answer
    assert(type(text_info[text_idx]['wrong_text']) == list)
    num_qs = len(text_info[text_idx]['wrong_text']) + 1
    if shuffle:
        np.random.seed(text_idx * 7)
        ordering = list(np.random.permutation(num_qs))
    else:
        ordering = None
        
    if not show_exp:
        # no explanation by setting all attr values to zero
        attr1 = np.zeros(1024)
        attr2 = [np.zeros(1024) for x in range(num_qs-1)]
        
        # vis = VisualizationDataRecord(
        #         text_info[text_idx]['random_attr_sent_viz']['correct'][0][0],
        #         text_info[text_idx]['random_attr_sent_viz']['correct'][0][0],
        #         [t[0] for t in text_info[text_idx]['random_attr_sent_viz']['wrong']],
        #         text_info[text_idx]['correct_summary'],
        #         text_info[text_idx]['original_text'],
        #         text_info[text_idx]['wrong_text'],
        #         text_info[text_idx]['random_attr_sent_viz']['correct'][0][1],
        #         attr1,
        #         attr2,
        #         text_info[text_idx]['score_t1_s1'],
        #         text_info[text_idx]['score_t2_s1'],
        #     )
        
        vis = VisualizationDataRecord(
                text_info[text_idx]['phrase_attrs']['summary']['joint_words'], 
                text_info[text_idx]['phrase_attrs']['correct']['joint_words'], 
                [
                    text_info[text_idx]['phrase_attrs']['wrong_1']['joint_words'], \
                    text_info[text_idx]['phrase_attrs']['wrong_2']['joint_words']
                ],
                text_info[text_idx]['correct_summary'], # raw summary text
                text_info[text_idx]['original_text'], # raw text
                text_info[text_idx]['wrong_text'], # raw list of wrong texts
                text_info[text_idx]['phrase_attrs']['summary']['joint_attrs'], 
                attr1,
                attr2,
                text_info[text_idx]['score_t1_s1'], # score for text1
                text_info[text_idx]['score_t2_s1'], # list of scores for text2
            )
    else:
        if exp_type == 'phrase' or exp_type == 'rouge_phrase': 
            # this is word/phrase + sentence level visualization
            vis = VisualizationDataRecord(
                text_info[text_idx]['%s_attrs'%exp_type]['summary']['joint_words'], 
                text_info[text_idx]['%s_attrs'%exp_type]['correct']['joint_words'], 
                [
                    text_info[text_idx]['%s_attrs'%exp_type]['wrong_1']['joint_words'], \
                    text_info[text_idx]['%s_attrs'%exp_type]['wrong_2']['joint_words']
                ],
                text_info[text_idx]['correct_summary'], # raw summary text
                text_info[text_idx]['original_text'], # raw text
                text_info[text_idx]['wrong_text'], # raw list of wrong texts
                text_info[text_idx]['%s_attrs'%exp_type]['summary']['joint_attrs'], 
                text_info[text_idx]['%s_attrs'%exp_type]['correct']['joint_attrs'], 
                [
                    text_info[text_idx]['%s_attrs'%exp_type]['wrong_1']['joint_attrs'], \
                    text_info[text_idx]['%s_attrs'%exp_type]['wrong_2']['joint_attrs']
                ],
                text_info[text_idx]['score_t1_s1'], # score for text1
                text_info[text_idx]['score_t2_s1'], # list of scores for text2
            )
        elif exp_type == 'lime': # lime
            # this is word-based visualization
            # post-process to nromalize the score values across the doocuments
            text_attr1 = text_info[text_idx]['lime_attr_words']['correct'][0][1]
            text_attr2 = [text_info[text_idx]['lime_attr_words']['wrong'][tt][1] for tt in range(2)]
            attr1_n = text_attr1.shape[0]
            attr2_n = text_attr2[0].shape[0]
            attr3_n = text_attr2[1].shape[0]
            tmp = np.hstack((text_attr1, text_attr2[0], text_attr2[1]))
            tmp = np.array(rescale_pos_neg_attrs(tmp))
            text_attr1 = tmp[:attr1_n]
            text_attr2 = [
                tmp[attr1_n:attr1_n+attr2_n],
                tmp[attr1_n+attr2_n:]
            ]
            
            vis = VisualizationDataRecord(
                text_info[text_idx]['%s_attr_words'%exp_type]['correct'][0][0],
                text_info[text_idx]['%s_attr_words'%exp_type]['correct'][0][0],
                [t[0] for t in text_info[text_idx]['%s_attr_words'%exp_type]['wrong']],
                text_info[text_idx]['correct_summary'],
                text_info[text_idx]['original_text'],
                text_info[text_idx]['wrong_text'],
                text_attr1,
                text_attr1,
                text_attr2,
                #text_info[text_idx]['%s_attr_words'%exp_type]['correct'][0][1],
                #text_info[text_idx]['%s_attr_words'%exp_type]['correct'][0][1],
                #[t[1] for t in text_info[text_idx]['%s_attr_words'%exp_type]['wrong']],
                text_info[text_idx]['score_t1_s1'],
                text_info[text_idx]['score_t2_s1'],
            )
        elif exp_type == 'presum':
            # show presum: this is sentence-based visualization
            vis = VisualizationDataRecord(
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][0],
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][0],
                [t[0] for t in text_info[text_idx]['%s_attr_sent_viz'%exp_type]['wrong']],
                text_info[text_idx]['correct_summary'],
                text_info[text_idx]['original_text'],
                text_info[text_idx]['wrong_text'],
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][1],
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][1],
                [t[1] for t in text_info[text_idx]['%s_attr_sent_viz'%exp_type]['wrong']],
                text_info[text_idx]['score_t1_s1'],
                text_info[text_idx]['score_t2_s1'],
            )
        elif exp_type == 'shap':
            # post-process to nromalize the score values across the doocuments
            text_attr1 = text_info[text_idx]['shap_attr_word']['correct'][0][1]
            text_attr2 = [text_info[text_idx]['shap_attr_word']['wrong'][tt][1] for tt in range(2)]
            attr1_n = text_attr1.shape[0]
            attr2_n = text_attr2[0].shape[0]
            attr3_n = text_attr2[1].shape[0]
            tmp = np.hstack((text_attr1, text_attr2[0], text_attr2[1]))
            tmp = np.array(rescale_pos_neg_attrs(tmp))
            text_attr1 = tmp[:attr1_n]
            text_attr2 = [
                tmp[attr1_n:attr1_n+attr2_n],
                tmp[attr1_n+attr2_n:]
            ]
            
            vis = VisualizationDataRecord(
                text_info[text_idx]['%s_attr_word'%exp_type]['correct'][0][0],
                text_info[text_idx]['%s_attr_word'%exp_type]['correct'][0][0],
                [t[0] for t in text_info[text_idx]['%s_attr_word'%exp_type]['wrong']],
                text_info[text_idx]['correct_summary'],
                text_info[text_idx]['original_text'],
                text_info[text_idx]['wrong_text'],
                text_attr1,
                text_attr1,
                text_attr2,
                text_info[text_idx]['score_t1_s1'],
                text_info[text_idx]['score_t2_s1'],
            )
        else:
            vis = VisualizationDataRecord(
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][0],
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][0],
                [t[0] for t in text_info[text_idx]['%s_attr_sent_viz'%exp_type]['wrong']],
                text_info[text_idx]['correct_summary'],
                text_info[text_idx]['original_text'],
                text_info[text_idx]['wrong_text'],
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][1],
                text_info[text_idx]['%s_attr_sent_viz'%exp_type]['correct'][0][1],
                [t[1] for t in text_info[text_idx]['%s_attr_sent_viz'%exp_type]['wrong']],
                text_info[text_idx]['score_t1_s1'],
                text_info[text_idx]['score_t2_s1'],
            )

    out, correct_answer = visualize(vis, exp_type, legend=legend, show_exp=show_exp, show_score=show_score, ordering=ordering)
    #out, correct_answer = visualize_text([vis], legend=legend, show_exp=show_exp, show_score=show_score, ordering=ordering)
    if none_of_above:
        correct_answer = -1
    return out, correct_answer+1
