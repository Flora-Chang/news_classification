#encoding: utf8

import http.client as httplib
import urllib
import random
import hashlib
import json

"""translate the ori chinese sentence to english, then english to chinese to increase diversity"""

def en_to_zh(src):
    """english to chinese"""

    appid = '20170226000039888'
    secretKey = 'JC611ckjXHMKrwwKqRhl'

    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = src
    fromLang = 'en'
    toLang = 'zh'
    salt = random.randint(32768, 65536)

    sign = appid + q + str(salt) + secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode("utf-8"))
    sign = m1.hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response is an object of HTTPResponse
        response = httpClient.getresponse()
        response1 = response.read()
        encoding = response.info().get_content_charset('utf8')  # JSON default
        data = json.loads(response1.decode(encoding))
        dst = data['trans_result'][0]['dst']
        #print(dst)
        return dst
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def zh_to_en(src):
    """chinese to english"""

    appid = '20170226000039888'
    secretKey = 'JC611ckjXHMKrwwKqRhl'

    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = src
    fromLang = 'zh'
    toLang = 'en'
    salt = random.randint(32768, 65536)

    sign = appid + q + str(salt) + secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode("utf-8"))
    sign = m1.hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response is an object of HTTPResponse
        response = httpClient.getresponse()
        response1 = response.read()
        encoding = response.info().get_content_charset('utf8')  # JSON default
        data = json.loads(response1.decode(encoding))
        dst = data['trans_result'][0]['dst']
        #print(dst)
        return dst
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()

def main():
    src_file = "../data/last_col.txt"
    dst_file = "../data/trans_corpus.txt"
    with open(src_file, 'r') as src_f, open(dst_file, 'w') as dst_f:
        for line in src_f:
            english = zh_to_en(line.strip())
            zhong = en_to_zh(english)
            dst_f.write(zhong+"\n")


if __name__ == '__main__':
    main()