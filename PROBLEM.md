# Problem Statement 

Step 1 - Create a Python script which performs basic RAG across the Hatchet documentation (https://docs.hatchet.run). This RAG pipeline should ingest and index all of our docs (if it's easier, you can also get them from here: https://github.com/hatchet-dev/hatchet/tree/main/frontend/docs). You can use any third-party service you need to, but please don't spend any personal money on this -- if you need access to something, we can set that up for you with a max budget of $10. 

Step 2 - write a FastAPI endpoint for submitting a question to the Hatchet docs which returns a list of relevant pages + paragraphs.

Step 3 - Create a writeup detailing how you built this and how to test it. Include a section on limitations with your approach and how you would improve it. Treat this like something that would go on the Hatchet blog! (example: https://docs.hatchet.run/blog/multi-tenant-queues). 

Bonus points if you use Hatchet for the RAG pipeline!