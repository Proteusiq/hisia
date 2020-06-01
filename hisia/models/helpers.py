import joblib
import re
import dill
import lemmy
from loguru import logger


STOP_WORDS = joblib.load('hisia/data/stops.pkl')
lemmatizer = lemmy.load('da')

@logger.catch
def tokenizer(blob, stop_words=STOP_WORDS, remove_digits=True):
    
    if stop_words is None:
        stop_words = {}
    
    blob = blob.lower()
    
     # eyes [nose] mouth | mouth [nose] eyes pattern
    emoticons = r"(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)"
    emoticon_re = re.compile(emoticons, re.VERBOSE | re.I | re.UNICODE)
    
    text = re.sub(r'[\W]+', ' ', blob)
    
    # remove 3+ repetitive characters i.e. hellllo -> hello, jaaaa -> jaa 
    repetitions = re.compile(r'(.)\1{2,}')
    text = repetitions.sub(r'\1\1', text)
    
    # remove 2+ repetitive words e.g. hej hej hej -> hej
    
    repetitions = re.compile(r'\b(\w+)\s+(\1\s*)+\b')
    text = repetitions.sub(r'\1 ', text)
    
    
    # 14år --> 14 år
    text = re.sub(r'([0-9]+(\.[0-9]+)?)', r' \1 ', text).strip()
    
    emoji = ''.join(re.findall(emoticon_re, blob))
    
       
    # remove stopwords
    text_nostop = [word for word in text.split() if word not in stop_words]
    
    # tokenization lemmatize
    lemmatized_text = [lemmatizer.lemmatize('', word)[-1]  
                                 for word in text_nostop]
    
    remove_stopwords = ' '.join(word for word in lemmatized_text if len(word)>1)
    
    if remove_digits:
        remove_stopwords = re.sub(r'\b\d+\b', '', remove_stopwords)
    

    # remove extra spaces
    remove_stopwords = ' '.join(remove_stopwords .split())
    result = f'{remove_stopwords} {emoji}'.encode('utf-8').decode('utf-8')
       
    
    return result.split()

@logger.catch
def persist_model(name,clf=None, method='load'):
    'Pass in the file name, object to be saved or loaded'
    import dill
    
    if method == 'load':
        with open(name,'rb') as f:
            return dill.load(f)
    elif method == 'save':
        logger.info(f'[+] Persisting {name} ...')
        if clf is None:
            raise ValueError('Pass Model/Pipeline/Transformation')
        with open(name,'wb') as f:
            dill.dump(clf,f)
            logger.info(f'[+] Persistence Complete. Model {name} is saved')
    else:
        raise ValeuError('Wrong arguments')

@logger.catch
def show_diagram(trained_clf, X_train, y_train, X_test, y_test, compare_test=True):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, classification_report
    
    print('Classification Report')
    print('\t','_'*45)
    print(classification_report(y_test,
             trained_clf.predict(X_test),target_names=['Negative','Positive']))
    
    print('\t','_'*45,'\n'*2)
    
    plt.figure(figsize=(10,5))
               
    title = 'Receiver operating characteristic'
    data = [[X_test, y_test, 'red','Test'],[X_train, y_train, 'blue','Train']]
    
        
    for i, j in enumerate(data):

        y_pred, y_pred_prob = trained_clf.predict(j[0]), trained_clf.predict_proba(j[0])[:,1]
        clf_score = trained_clf.score(j[0], j[1])

        fpr,tpr,_ = roc_curve(j[1], y_pred_prob) # remember we need binary

        plt.plot(fpr,tpr,lw=4, 
                 color=j[2], label='{} ROC curve (area ={:.2f})'.format(j[3], clf_score));
        
        if not compare_test:
            break
       
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}'.format(title))
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('visualization/ROC.png')

# Function modification of Mike Lee Williams(mike@mike.place)
def show_most_informative_features(feature_names, clf, n=1000):
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % ((coef_1), fn_1, (coef_2), fn_2))