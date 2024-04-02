css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.css-ffhzg2 {
    position: absolute;
    background: rgb(0 0 0);
    color: rgb(250, 250, 250);
    inset: 0px;
    overflow: hidden;
}
.st-emotion-cache-uf99v8{
    background: black;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/a9/4a/ee/a94aee835e16cff4f14c83dac8ffbe10.gif" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://fiverr-res.cloudinary.com/videos/so_0.379904,t_main1,q_auto,f_auto/yqkqrmhjwpqqgznzmpen/create-your-personalized-animated-talking-avatar-using-powerful-ai.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
