import os
from io import BytesIO

import discord
import requests
from PIL import Image

from captioning import Pipeline

client = discord.Client()
pipe = Pipeline()


def prepare_attachment(im, filename, resize=False):
    if resize:
        im = im.resize((256, 256))  # Super lazy, consider rescaling
    temp = BytesIO()
    temp.name = filename
    im.save(temp, format='PNG')
    temp.seek(0)
    return discord.File(temp, filename=temp.name)


@client.event
async def on_message(message):
    await client.wait_until_ready()
    if message.author.id == client.user.id:
        return
    if message.attachments:
        attachment = message.attachments[0]

        im = Image.open(requests.get(attachment.url, stream=True).raw)
        caption = pipe(im)

        try:
            await message.channel.send(caption, file=prepare_attachment(im, attachment.filename))
            await message.delete()
        except discord.errors.HTTPException:
            await message.channel.send(caption, file=prepare_attachment(im, attachment.filename, resize=True))
            await message.delete()
        except Exception as e:
            await message.channel.send(f'Error: {e}')
    else:
        await message.channel.send("No attachments to be found...Can't caption dat! Try sending me an image 😉")


client.run(os.environ.get("DISCORD_TOKEN", None))
