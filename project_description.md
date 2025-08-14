Crafting an AI-Powered HR Assistant: A Use Case for Nestle’s HR Policy Documents


Master Prompt:

Create an AI-Powered HR Assistant chatbot. Please provide step by step instructions for the entire workflow: setting up the programming environment, processing text documents, creating text vector representations, and building a question-answering system. 

Overview:
The project aims to create a conversational chatbot that responds to user inquiries using PDF document information. It requires proficiency in extracting and converting text into numerical vectors, establishing an answer-finding mechanism, and designing a user-friendly chatbot interface with Flask. Additionally, the initiative emphasizes structuring inquiries for clear communication and deploying the chatbot for practical use, guaranteeing the system's accessibility and efficiency in meeting user needs.

Instructions:
• Read the sections on situation, task, action, and result carefully to understand the 
assignment. 
• Adhere closely to the provided guidelines, ensuring your submission contains all 
necessary analyses and interpretations. 


Context:
As a developer, you have received the critical task of improving the operational efficiency of Nestlé's human resources department, a leading multinational corporation. Your toolkit 
includes cutting-edge conversational AI technology, Python libraries, LangChain, the powerful GPT model from OpenAI, and the user-friendly Flask UI. Your mission is to integrate these advanced tools seamlessly to transform HR processes, creating a more streamlined and efficient workflow within the Nestlé organization. 

Task:
Your task is to develop a conversational chatbot. This chatbot must answer queries about 
Nestlé's HR reports efficiently. Use Python libraries, LangChain, OpenAI's GPT model, and Flask UI. 
These tools will help you create a user-friendly interface. This interface will extract and 
process information from documents. It will provide accurate responses to user queries. 

Action:
• Import essential tools and set up OpenAI's API environment. 
• Load Nestle's HR policy pdf (./the_nestle_hr_policy_pdf_2012.pdf) using PyPDFLoader and split it in chunks for easy processing. 
• Create vector representations for text chunks using FAISS and OpenAI's 
embeddings. 
• Build a question-answering system using the gpt-4o-mini model to retrieve answers 
from text chunks. 
• Create a prompt template to guide the chatbot in understanding and responding to 
users. 
• Use Flask to build a modern and beautiful chatbot interface, enabling interaction and 
information retrieval. Ensure the interface is user-friendly to facilitate effective interaction and information retrieval. 

