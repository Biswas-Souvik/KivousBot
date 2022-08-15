import logging, psycopg2
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from chat import get_response


TOKEN = '<my telegram token>'
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
print("Bot started...")

DATABASE_URL = 'postgres://kvmkncdtlrzgey:72205a5bb789a9e501a973ac0b9aecb5843db2f5790cd4914952586bab27f778@ec2-34-193-44-192.compute-1.amazonaws.com:5432/d8hb9hgmg5r0l9'
msg_db = psycopg2.connect(DATABASE_URL, sslmode='require')
#  create a new cursor
mycursor = msg_db.cursor()
mycursor.execute('''CREATE TABLE IF NOT EXISTS messages (
    user_id varchar(50),
    first_name varchar(50),
    user_name varchar(50),
    message varchar(200)
    )'''
)
mycursor.execute('''ALTER TABLE messages ADD COLUMN IF NOT EXISTS reply varchar(200)''')


async def start_function(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_text = '''Hi, it's me your friendly neighbor kivous bot.😊
I am learning english as my first human language to connect with you guys.
let chat! I will store your messages and try to learn from them.
happy learning to me :D 
try asking for help by clicking here /help'''
    await context.bot.send_message(chat_id=update.effective_chat.id, text=start_text)


async def help_function(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """I am still in learning phase but I have two jokes for you.
I will tell you if you say 'tell me a joke' """
    await context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = str(update.message.text)
    response = get_response(text.lower())
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    
    first = str(update.effective_chat.first_name)
    id = str(update.effective_chat.id)
    name = str(update.effective_chat.username)
    print(id, first, name, text, response)
    
    query = 'INSERT INTO messages (user_id,first_name,user_name, message, reply) VALUES (%s, %s, %s, %s, %s)'
    data = (id, name, first, text, response)
    
    mycursor.execute(query, data)
    msg_db.commit()


if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start_function))
    
    application.add_handler(CommandHandler('help', help_function))

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
   
    application.run_polling()
