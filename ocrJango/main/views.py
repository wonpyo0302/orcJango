from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
import json
from openai import OpenAI
import os


# Create your views here.
def helloworld(request):
    return render(request, 'ocrJango/helloworld.html', {})


@require_POST
def checkdata(request):
    client = OpenAI(
        api_key = os.environ["OPENAI_API_KEY"]
    )
    jsonObj = json.loads(request.body)

    messages = [
        {"role": "system", "content": "You are a helpful assistant in the summary"},
        {"role": "user", "content": f"Summarize the following. \n {jsonObj.get('contents')}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=4000,
        temperature=0.8,
        n=1
    )

    resData = {
        'result':response.choices[0].message.content
    }

    return JsonResponse(resData)
