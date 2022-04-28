import requests, json

headers = {"User-agent": "your bot 0.1"}


def get_value(l, m):
    for i in l:
        try:
            m = m[i]
        except:
            return ""
    return m


def clean_transcript(transcript):
    transcript_list = transcript.split("\n")
    filtering = (
        lambda x: x.find("-->") == -1
        and len(x)
        and x not in ["(Laughter)", "(Applause)", "WEBVTT"]
    )
    transcript_list_filtering = filter(filtering, transcript_list)
    return " ".join(transcript_list_filtering)


def get_transcript(url):
    start_string = "project_masters/"
    start_index = url.find(start_string) + len(start_string)
    end_index = url[start_index:].find("/")

    id = url[start_index : start_index + end_index]
    url = f"https://hls.ted.com/project_masters/{id}/subtitles/en/full.vtt"

    transcript_res = requests.get(url, headers)
    if transcript_res.status_code == 200:
        return clean_transcript(transcript_res.text)
    return None


def get__json_obj(url):
    res = requests.get(url.strip(), headers)
    html = res.text
    start_string = '<script id="__NEXT_DATA__" type="application/json">'
    start_index = html.find(start_string) + len(start_string)
    end_index = html[start_index:].find("</script>")
    json_obj = html[start_index : start_index + end_index]
    return json_obj


def repeated_request(url, count):
    if count == 0:
        raise Exception("ERROR:" + url)
    json_obj = get__json_obj(url)
    if json_obj:
        repeated_request(url, count - 1)
    return json_obj


def download_entity(urls, id_, csv_list):
    for url in urls:
        try:
            json_obj = repeated_request(url, 3)
            print(json_obj)
        except:
            print("ERROR:" + url)
        else:
            metadata = json.loads(json_obj)["props"]["pageProps"]

            d = dict()

            d["talk__id_text"] = id_
            d["talk__id"] = get_value(["videoData", "id"], metadata)
            d["talk__name"] = get_value(["videoData", "title"], metadata)
            d["talk__description"] = get_value(["videoData", "description"], metadata)

            try:
                playerData = json.loads(
                    get_value(["videoData", "playerData"], metadata)
                )
                metadata_url = get_value(["resources", "hls", "metadata"], playerData)
                if metadata_url:
                    d["transcript"] = get_transcript(metadata_url)
            except:
                d["transcript"] = None

            csv_list.append(d)
