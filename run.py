import os
import time
import tomd
import requests

from bs4 import BeautifulSoup

BaseUrl = r'https://paperswithcode.com'


def crawl_method_category_url(url):

    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    ret = {}
    for a in soup.find_all("h4", attrs={"style": "padding-bottom:25px;"}):
        c = a.find('a')
        ret[c.string] = BaseUrl + c.get('href')
    return ret


def crawl_area_category_url(url):

    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    ret = {}
    for a in soup.find_all("h4", attrs={"style": "padding-bottom:25px;"}):
        c = a.find('a')
        ret[c.string] = BaseUrl + c.get('href')
    return ret


def crawl_methods_url(url):

    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    ret = {}
    table = soup.find("table", attrs={"id": "methodsTable"})
    for idx, tr in enumerate(table.find_all('tr')):
        if idx != 0:
            c = tr.find('td').find('a')
            ret[c.text.strip()] = 'https://paperswithcode.com' + c.get('href')
    return ret


def get_content(img_dir, url):

    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    a = soup.find("div", attrs={"class": "col-md-8 description"})
    img_url = soup.find("img", attrs={"id": "imageresource"})
    source_url = soup.find("span", attrs={"class": "description-source"})

    s = ''
    if img_url:
        img_src = img_url['src']
        img_name = img_src.split('/')[-1]
        img_path = '{}/{}'.format(img_dir, img_name)
        if not os.path.exists(img_path):
            with open(img_path, 'wb') as imgf:
                imgf.write(requests.get(BaseUrl+'/'+img_src).content)
        s += '![](./img/{})'.format(img_name)+'\n'
    s += tomd.Tomd(str(a)).markdown+'\n'
    if source_url:
        source_url = source_url.find('a').get('href')
        s += 'source: [source]({})\n'.format(source_url)
    return s


if __name__ == '__main__':

    method_dict = crawl_method_category_url(BaseUrl+'/methods')
    # https://paperswithcode.com/methods/area/general

    for name, url in method_dict.items():   # General
        if not os.path.exists(name):
            os.makedirs(name)
        img_dir = f'./{name}/img'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        area_dict = crawl_area_category_url(url)
        # https://paperswithcode.com/methods/category/skip-connection-blocks

        for _name, _url in area_dict.items():  # Skip Connection Blocks
            file_path = './{}/{}.md'.format(name, _name.replace(' ', '_'))

            methods_dict = crawl_methods_url(_url)
            with open(file_path, 'a') as f:
                for __name, __url in methods_dict.items():
                    print(__name)
                    f.write(f'# [{__name}]({__url})\n')
                    ctt = get_content(img_dir, __url)
                    f.write(ctt)
                    time.sleep(6)
