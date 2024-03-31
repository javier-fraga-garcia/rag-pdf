from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter
from warnings import filterwarnings

from utils.load import load_documents
from utils.retriever import get_retriever

filterwarnings('ignore')

def create_chain():
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    documents = load_documents('./docs/Criminal_Intelligence_for_Analysts.pdf', splitter=doc_splitter)
    retriever = get_retriever(documents)
    template = """
        Vas a actuar como un asistente virtual para pregutnas y respuestas. Te voy a facilitar una cierta información de contexto que vas a usar para responder a la pregunta formulada.
        Tienes que ser preciso en la respuesta aunque puedes adoptar un ligero tono ironico o hacer bromas. Si no encuentras información para responder a la pregunta responderás "Lo siento, no puedo ayudarte con eso".
        Siempre responderas en texto como si fueras una persona. No antepondrás a la respuesta expresiones como "Respuesta:" o indicaciones similares.

        Contexto: {context}
        Pregunta: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    
    model = ChatOpenAI(temperature=0.7)

    chain = (
        {'context': itemgetter('question') | retriever, 'question': itemgetter('question')} |
        prompt | model | parser
    )
    return chain

def main():
    chain = create_chain()
    print('Bienvenido al chat')
    print('-'*20)
    while True:
        question = input('Pregunta: ')
        if question == 'exit':
            break
        print(chain.invoke({'question': question}))
        print('\n')


if __name__ == '__main__':
    main()