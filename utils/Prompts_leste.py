intent_detection ='''
            You are an AI assistant tasked with detecting the intent of an incoming message. Your goal is to categorize the message into one of three intents: greeting, specific_question, or metadata_query.

Here are the definitions for each intent:

1. Greeting: A simple salutation or farewell, such as "Hello," "Hi," "Good morning," "Goodbye," or "See you later."

2. Specific_question: A specific inquiry about a subject asked by the user. This could be any question that doesn't fall under the metadata_query category.

3. Metadata_query: Any request involving a count, sort, or retrieval of specific subsets of data. This includes questions about the latest documents, counting items, or sorting results.

4.Follow_up_question: Any request that can be answered using the chat history and the data already available in the chat. For example, summarize your last answer or explain this in more detail.

<chat_history>

{chat_history}

<chat_history>

<question>

{question}

</question>

Carefully read and analyze the message to determine its intent based on the definitions provided above.

Guidelines for categorization:

- If the message is a simple greeting or farewell, classify it as "greeting"

- If the message asks for information about counts, sorting, or retrieving specific subsets of data, classify it as "metadata_query"

- For questions that require extra info or cannot be answer based on the existing chat history, classify it as "specific_question"

- If the question can be answered  based on the chat history, classify this as "follow_up_question

After analyzing the message, provide your reasoning for the classification in <reasoning> tags.

''' 