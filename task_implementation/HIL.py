from langgraph.types import interrupt, Command

def human_node(state: State):
    value = interrupt( 
        {
            "text_to_revise": state["some_text"] 
        }
    )
    return {
        "some_text": value 
    }


graph = graph_builder.compile(checkpointer=checkpointer) 

# Run the graph until the interrupt is hit.
config = {"configurable": {"thread_id": "some_id"}}
result = graph.invoke({"some_text": "original text"}, config=config) 
print(result['__interrupt__']) 
# > [
# >    Interrupt(
# >       value={'text_to_revise': 'original text'}, 
# >       resumable=True,
# >       ns=['human_node:6ce9e64f-edef-fe5d-f7dc-511fa9526960']
# >    )
# > ] 

print(graph.invoke(Command(resume="Edited text"), config=config)) 
# > {'some_text': 'Edited text'}