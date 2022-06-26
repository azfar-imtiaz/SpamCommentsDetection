import os
import pickle
import urllib.parse as p
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

from globals import PATH_TO_CREDENTIALS_FILE


class YouTubeCommentExtractor:
    """
        This implementation has been heavily inspired from a tutorial.
        Link: https://www.thepythoncode.com/code/using-youtube-api-in-python
    """
    def __init__(self):
        """
            Constructor for YouTubeCommentExtractor class.
        """
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.client_secrets_file = PATH_TO_CREDENTIALS_FILE
        self.token_file_name = "token.pickle"
        self.SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
        self.youtube = self.__youtube_authenticate()

    def __youtube_authenticate(self):
        """
            YouTube authentication function. Requires credentials.json from YouTube Console API.
        """
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        creds = None
        # the file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time
        if os.path.exists(self.token_file_name):
            with open(self.token_file_name, "rb") as token:
                creds = pickle.load(token)
        # if there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            # save the credentials for the next run
            with open(self.token_file_name, "wb") as token:
                pickle.dump(creds, token)

        return build(self.api_service_name, self.api_version, credentials=creds)

    @staticmethod
    def get_video_id_by_url(url):
        """
        Return the Video ID from the video `url`
        """
        # split URL parts
        parsed_url = p.urlparse(url)
        # get the video ID by parsing the query of the URL
        video_id = p.parse_qs(parsed_url.query).get('v')
        if video_id:
            return video_id[0]
        else:
            raise Exception(f"Wasn't able to parse video URL: {url}")

    @staticmethod
    def parse_channel_url(url):
        """
        This function takes channel `url` to check whether it includes a
        channel ID, user ID or channel name
        """
        path = p.urlparse(url).path
        id = path.split("/")[-1]
        if "/c/" in path:
            return "c", id
        elif "/channel/" in path:
            return "channel", id
        elif "/user/" in path:
            return "user", id

    def __get_comments(self, **kwargs):
        return self.youtube.commentThreads().list(
            part="snippet",
            **kwargs
        ).execute()

    def get_comments_from_youtube_video(self, url, num_pages=50):
        # URL can be a video, to extract comments
        if 'watch' in url:
            # that's a video
            video_id = self.get_video_id_by_url(url)
            params = {
                'videoId': video_id,
                'maxResults': 20,
                # 'order': 'relevance',  # default is 'time' (newest)
            }
        else:
            raise ValueError("Please specify a valid YouTube video link!")

        video_comments = []
        # get the first num_pages (num_pages API requests)
        print("Scraping comments...")
        for index in range(num_pages):
            print("\tCurrent page: {}".format(index + 1))
            # make API call to get all comments from the channel (including posts & videos)
            response = self.__get_comments(**params)
            items = response.get('items')
            # if items is empty, break out of the loop
            if not items:
                break
            for item in items:
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                # updated_at = item['snippet']['topLevelComment']['snippet']['updatedAt']
                # like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
                comment_id = item["snippet"]["topLevelComment"]['id']
                author_name = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comment_date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                video_comments.append({
                    'comment_text': comment_text,
                    'comment_id': comment_id,
                    'author': author_name,
                    'date': comment_date
                })
            if 'nextPageToken' in response:
                # if there is a next page, add next page token to the params we pass to the function
                params['pageToken'] = response['nextPageToken']
            else:
                # must be end of comments
                break
        return video_comments
