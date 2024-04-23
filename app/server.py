from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.chain import chain as rar_chatbot
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rar_chatbot, path="/rar-chatbot")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
