
from datetime import date

REFERENCE_DATE = date.today()  

AI_ML_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "deep learning",
    "ai engineer", "artificial intelligence",
    "data scientist", "data science",
    "nlp", "natural language",
    "search engineer", "ranking", "recommendation",
    "retrieval", "information retrieval",
    "applied scientist",
]

NON_TECHNICAL_TITLES = [
    "hr manager", "human resources",
    "marketing manager", "marketing executive", "marketing director",
    "sales executive", "sales manager", "sales director",
    "accountant", "accounting manager",
    "graphic designer", "ui designer", "ux designer", "web designer",
    "content writer", "content manager", "copywriter",
    "operations manager", "operations director",
    "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "administrative", "receptionist", "clerk",
    "recruiter", "talent acquisition",
]

CV_SPEECH_ROBOTICS_TITLES = [
    "computer vision", "cv engineer", "image processing",
    "speech engineer", "speech recognition", "asr engineer",
    "robotics", "robot engineer", "autonomous",
    "perception engineer", "3d vision", "image recognition",
]

JUNIOR_TITLES = [
    "junior", "intern", "trainee", "fresher", "associate",
]

CAREER_DESCRIPTION_KEYWORDS = {
    "core_retrieval_ranking": {
        "weight": 3.0,
        "keywords": [
            "embeddings", "embedding", "vector database", "vector search",
            "vector store", "retrieval", "ranking", "re-ranking", "reranking",
            "recommendation system", "recommendation engine", "recommender",
            "recommender system", "collaborative filtering",
            "search engine", "search ranking", "search relevance",
            "bm25", "tf-idf", "tfidf",
            "faiss", "elasticsearch", "elastic search", "opensearch",
            "pinecone", "qdrant", "milvus", "weaviate", "chroma",
            "annoy", "hnsw", "approximate nearest neighbor",
            "semantic search", "hybrid search",
            "candidate retrieval", "document retrieval", "vector db",
            "embedding space", "nearest neighbor search", "similarity search",
            "content-based filtering", "content based filtering",
            "query understanding", "intent classification", "relevance scoring",
            "re-ranker", "reranker", "click-through rate", "ctr optimization",
            "dense retrieval", "sparse retrieval", "two-tower", "cross-encoder",
            "bi-encoder", "knn", "nearest neighbor",
            "search system", "search infrastructure", "search platform",
            "search backend", "search service",
            "discovery engine", "discovery system", "discovery platform",
            "matching engine", "matching system", "matching algorithm",
            "talent matching", "candidate matching", "job matching",
            "personalization engine", "personalization system",
            "feed ranking", "content ranking",
            "inverted index", "posting list",
            "query expansion", "query rewriting", "query suggestion",
            "position bias", "selection bias", "exposure bias",
            "implicit feedback", "explicit feedback", "click feedback",
            "cold start", "cold-start",
            "multi-armed bandit", "explore exploit",
            "item recommendation", "product recommendation",
            "candidate generation", "candidate selection",
            "recall stage", "retrieval stage", "ranking stage",
            "pgvector",
        ],
    },
    "eval_rigor": {
        "weight": 2.5,
        "keywords": [
            "ndcg", "mrr", "mean reciprocal rank", "mean average precision",
            "a/b test", "ab test", "a/b experiment",
            "precision", "recall", "f1 score", "f1-score",
            "offline evaluation", "online evaluation",
            "benchmark", "evaluation framework", "eval pipeline",
            "learning to rank", "learning-to-rank",
            "hit rate", "recall@", "precision@", "coverage", "diversity",
            "interleaving", "counterfactual", "uplift",
            "ground truth", "annotation", "labeling", "labelling",
            "human evaluation", "human judgment",
            "statistical significance",
            "conversion rate", "click rate",
            "engagement metric", "retention metric",
            "holdout", "cross-validation", "cross validation",
            "online experiment", "controlled experiment",
            "regression test", "quality assurance",
        ],
    },
    "production_evidence": {
        "weight": 2.0,
        "keywords": [
            "deployed", "deploying", "deployment",
            "production", "prod environment",
            "scaled", "scaling", "scale up", "scale out",
            "million users", "thousands of users", "real users",
            "latency", "p99", "p95", "response time",
            "inference", "model serving", "serving infrastructure",
            "real-time", "realtime", "real time",
            "microservice", "api endpoint", "rest api", "grpc",
            "kubernetes", "k8s", "docker",
            "ci/cd", "mlops", "ml pipeline",
            "monitoring", "observability", "alerting",
            "load balancer", "rate limiting", "caching", "redis",
            "feature store", "model registry", "canary deployment",
            "traffic", "qps", "queries per second", "requests per second",
            "uptime", "sla",
            "rollback", "blue-green", "shadow mode",
            "batch inference", "online inference",
            "model versioning", "model lifecycle",
            "data pipeline", "feature pipeline",
            "throughput", "concurrency",
        ],
    },
    "llm_finetuning": {
        "weight": 2.0,
        "keywords": [
            "llm", "large language model",
            "rag", "retrieval augmented generation", "retrieval-augmented",
            "fine-tuning", "fine tuning", "finetuning",
            "lora", "qlora", "peft",
            "transformer", "attention mechanism",
            "embedding model", "sentence transformer",
            "huggingface", "hugging face",
            "prompt engineering", "prompt tuning",
            "langchain", "llamaindex", "trained llms", "trained llm",
            "gpt", "bert", "t5", "llama",
            "instruction tuning", "preference tuning", "rlhf", "dpo",
            "mistral", "gemma", "phi",
        ],
    },
    "python_depth": {
        "weight": 1.5,
        "keywords": [
            "python", "fastapi", "flask", "django",
            "async", "asyncio", "multiprocessing", "multithreading",
            "pytest", "unit test", "integration test",
            "code review", "pull request",
            "numpy", "pandas", "scikit-learn", "sklearn",
            "pytorch", "tensorflow", "keras", "jax",
            "spark", "pyspark",
        ],
    },
}

MUST_HAVE_SKILLS = [
    "python", "embeddings", "vector database", "vector search",
    "ranking", "recommendation", "retrieval",
    "ndcg", "mrr", "map", "a/b testing",
    "nlp", "natural language processing",
    "machine learning", "deep learning",
]

NICE_TO_HAVE_SKILLS = [
    "llm", "fine-tuning", "finetuning", "lora", "peft",
    "learning to rank", "learning-to-rank",
    "hr-tech", "recruiting", "talent",
    "distributed systems", "distributed computing",
    "open source", "open-source",
    "rag", "retrieval augmented generation",
    "transformer", "bert", "gpt",
    "fastapi", "flask",
    "docker", "kubernetes",
    "elasticsearch", "faiss", "pinecone", "qdrant", "milvus",
]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "cap gemini",
    "deloitte", "hcl technologies",
    "tech mahindra", "l&t infotech", "ltimindtree",
    "mindtree", "mphasis", "hexaware", "persistent systems",
    "zensar", "cyient", "sonata software",
    "genpact",                    
    "ey", "ernst & young", "ernst and young",
    "kpmg",
    "ibm consulting",
    "dxc technology", "dxc",
    "birlasoft", "larsen & toubro",
    "mu sigma", "fractal analytics", "tiger analytics",
    "latentview", "latentview analytics",
    "absolutdata", "bridgei2i",
]

RESEARCH_ONLY_TITLES = [
    "research scientist", "research fellow", "research engineer",
    "postdoc", "post-doc", "postdoctoral",
    "phd student", "phd candidate", "doctoral",
    "professor", "lecturer", "faculty",
]

EXPERIENCE_SWEET_SPOT = (5, 9)

PREFERRED_CITIES = [
    "pune", "noida", "hyderabad", "mumbai",
    "delhi", "new delhi", "gurgaon", "gurugram",
    "ncr", "bangalore", "bengaluru",
]

LOCATION_SCORES = {
    "india_preferred_city":        1.00,
    "india_other_relocate":        0.85,
    "india_other_no_relocate":     0.60,
    "outside_india_relocate":      0.40,
    "outside_india_no_relocate":   0.15,
}

BEHAVIORAL_WEIGHTS = {
    "open_to_work":             0.20,
    "last_active_recency":      0.15,
    "recruiter_response_rate":  0.15,
    "notice_period":            0.12,
    "interview_completion":     0.08,
    "github_activity":          0.08,
    "extra_signals":            0.22,   
}

OPEN_TO_WORK_SCORE = {True: 1.0, False: 0.70}  

LAST_ACTIVE_THRESHOLDS = [
    (30, 1.00),    
    (90, 0.80),    
    (180, 0.50),   
    (9999, 0.20),  
]

NOTICE_PERIOD_THRESHOLDS = [
    (30, 1.00),    
    (60, 0.90),    
    (90, 0.75),    
    (120, 0.50),   
    (9999, 0.30),  
]

INTERVIEW_COMPLETION_THRESHOLDS = [
    (0.40, 0.50),  
    (0.70, 0.80),  
    (1.01, 1.00),  
]

GITHUB_ACTIVITY_THRESHOLDS = [
    (-1, 0.65),    
    (0, 0.65),     
    (20, 0.80),    
    (40, 0.90),    
    (101, 1.00),   
]

HONEYPOT_TIMELINE_RATIO = 2.0  

EDUCATION_TIER_BONUS = {
    4: 0.12,   
    3: 0.08,   
    2: 0.04,   
    1: 0.00,   
    0: 0.00,   
}
RELEVANT_DEGREE_MULTIPLIER = 1.5

RELEVANT_ASSESSMENT_KEYWORDS = [
    "python", "ml", "machine learning", "data", "algorithm",
    "ai", "deep learning", "nlp", "statistics",
]

SALARY_CONCERN_THRESHOLD_LPA = 60
SALARY_RED_FLAG_THRESHOLD_LPA = 80

CAREER_TITLE_WEIGHT = 0.35    
CAREER_DESC_WEIGHT = 0.65     

MULTIPLICATIVE_FLOORS = {
    "skills":       0.30,   
    "experience":   0.40,   
    "location":     0.25,   
    "behavioral":   0.35,   
}
CAREER_EXPONENT = 1.3              
EDUCATION_ADDITIVE_BONUS = 0.02    
TFIDF_ADDITIVE_BONUS = 0.03        
QUANTITATIVE_ADDITIVE_BONUS = 0.03 
