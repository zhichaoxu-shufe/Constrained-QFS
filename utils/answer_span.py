from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline

class AnswerSpanPipeline:
    def __init__(self):
        self.model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
        self.pipeline = pipeline(
            'question-answering', 
            model=self.model_name,
            tokenizer=self.model_name
            )
    
    def get_answer_span(self, query, context):
        QA_input = {
            'question': query,
            'context': context,
        }
        # print('QA input: ', QA_input)
        output = self.pipeline(QA_input)
        # print('output: ', output)
        return output['answer']