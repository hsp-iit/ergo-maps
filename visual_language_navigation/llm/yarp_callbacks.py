import yarp

class ChatCallbck(yarp.BottleCallback):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def onRead(self, bot, reader):
        try:
            chat_string = bot.get(0).asString()

            self.wrapper.chat(chat_string)
        except Exception as ex:
            print(f"chat_failed {ex=}")
