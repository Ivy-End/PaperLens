import os
import logging
import requests
import numpy as np

from .Embedder import Embedder
from .AIClient import GeminiClient
from .MarkdownRenderer import MarkdownRenderer
from .Mailer import Mailer
from .FetchPaper.Aggregator import Aggregator

from .FetchPaper.ArxivSource import ArxivSource
from .FetchPaper.CORESource import CORESource
from .FetchPaper.CrossrefSource import CrossrefSource
from .FetchPaper.DBLPSource import DBLPSource

from .FetchPaper.PubMedSource import PubMedSource
from .FetchPaper.OpenAlexSource import OpenAlexSource

log = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.embedder = Embedder(config.EMBEDDING_MODEL)
        self.renderer = MarkdownRenderer()
        # self.ai = GeminiClient(config.GEMINI_KEY, config.GEMINI_MODEL) if (config.AI_ENABLE and config.GEMINI_KEY) else None
        self.mailer = Mailer(config.EMAIL_SERVER, config.EMAIL_PORT)
        
        self.aggregator = Aggregator([
            ArxivSource(),
            CrossrefSource(),
        ])

    def Run(self, *, day : str, nextDay : str):

        log.info(f'Pipeline started for day: {day}')

        # 1) Zotero 用户画像
        log.info(f'Fetching user profile from Zotero...')
        personasTexts = []
        zoteroUser = os.getenv("ZOTERO_USER")
        zoteroKey  = os.getenv("ZOTERO_KEY")
        headers = {"Zotero-API-Key": zoteroKey}
        baseUrl = f"https://api.zotero.org/users/{zoteroUser}/items?format=json&limit=9999&sort=dateModified&direction=desc"
        log.info(f'{baseUrl}')
        
        zoteroPapers = requests.get(baseUrl, headers = headers, timeout = 60).json()
        totalPapers = 0
        for paper in zoteroPapers:
            if "data" in paper:
                dataField = paper["data"]
                if "title" in dataField and "abstractNote" in dataField:
                    totalPapers += 1
                    personasTexts.append((f"## 论文 {totalPapers}\n- 标题：" + dataField["title"] + "\n- 摘要：" + dataField["abstractNote"]).strip())
                    log.info(f"- Loaded paper from Zotero ({totalPapers}): " + dataField["title"])

        # 2) 文本嵌入
        log.info(f'Embedding user profile texts...')
        embeddings = self.embedder.Encode(personasTexts)
        if embeddings.size == 0:
            personasVecs = np.zeros((1, self.embedder.dimensions), dtype = np.float32)
        else:
            personasVecs = embeddings.mean(axis = 0, keepdims = True)
            personasVecs /= (np.linalg.norm(personasVecs) + 1e-9)

        # 3) 抓取候选论文 + 嵌入
        log.info(f'Fetching candidate papers for {day}...')
        rawDataset = self.aggregator.fetch_all(day=day, nextDay=nextDay, **{"OpenAlex":{"perPage":200,"maxPages":6}})

        totalPapers = 0
        paperTexts = []
        paperCandidates = []
        for rawPaper in rawDataset:
            title        = rawPaper.get("title", "") or ""
            abstractNote = rawPaper.get("abstract", "") or ""
            if title.strip() == "" and abstractNote.strip() == "":
                continue
            
            totalPapers += 1
            paperCandidates.append(rawPaper)
            paperTexts.append(f"## 论文 {totalPapers}\n- 标题：{title}\n- 摘要：{abstractNote}")
        
        embeddings = self.embedder.Encode(paperTexts)
        
        # 4) 相似度打分
        log.info(f'Ranking candidate papers...')
        paperSimilarity = (embeddings @ personasVecs.T).ravel() if embeddings.size else np.zeros((0, ), dtype = np.float32)
        rankOrder = np.argsort(paperSimilarity)[::-1][:self.config.TOP_K]
        paperRecommendations = []
        for index in rankOrder:
            paper = {**paperCandidates[index]}
            paper["Similarity"]=float(paperSimilarity[index])
            paper.setdefault("abstract", paper.get("abstract", ""))
            paperRecommendations.append(paper)

        # 5) （可选）Gemini 摘要/理由
        # if self.ai and picks:
        #     picks = self.ai.summarize_batch(picks, personasNote)

        # 6) 渲染 + 邮件
        log.info(f'Rendering markdown and sending email...')
        markdown = self.renderer.Render(day, paperRecommendations)
        self.mailer.SendMarkdown(subject=f"[PaperLens] {day}", markdownText = markdown)
