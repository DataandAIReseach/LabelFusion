"""Collect the LLM expert's votes for every dated sentence, using verbatim the prompt that
Shah et al. (2023) used for their ChatGPT baseline. Resumable; atomic writes.

Works with any OpenAI-compatible chat endpoint. Configure via environment:
    LLM_API_URL   (default https://ollama.com/v1/chat/completions)
    LLM_API_KEY   (required)
    LLM_MODEL     (default gemma4:31b)

    LLM_API_KEY=... python scripts/2_collect_votes.py
"""
import asyncio, hashlib, json, os, time

import aiohttp
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.environ.get("LLM_API_URL", "https://ollama.com/v1/chat/completions")
KEY = os.environ.get("LLM_API_KEY") or exit("set LLM_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "gemma4:31b")
CONC, CHUNK = 8, 120

PROMPT = ("Discard all the previous instructions. Behave like you are an expert sentence "
          "classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or "
          "'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary "
          "policy, 'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' "
          "if the stance is neutral. Provide the label in the first line and provide a short "
          "explanation in the second line. The sentence: {s}")

key = lambda t: hashlib.sha1(("authors-verbatim-v1|" + str(t)).encode()).hexdigest()


async def main():
    df = pd.read_csv(f"{ROOT}/data/gold_dated.csv")
    sents = df["sentence"].astype(str).tolist()
    OUTF = f"{ROOT}/data/llm_votes_fomc.json"
    done = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
    todo = [s for s in sents if key(s) not in done]
    print(f"{len(done)} cached, {len(todo)} to fetch")
    sem = asyncio.Semaphore(CONC)

    async def call(sess, prompt):
        for _ in range(5):
            try:
                async with sem, sess.post(API, headers={"Authorization": f"Bearer {KEY}"},
                                          json={"model": MODEL, "temperature": 0,
                                                "messages": [{"role": "user", "content": prompt}]},
                                          timeout=aiohttp.ClientTimeout(total=300)) as r:
                    if r.status == 429:
                        await asyncio.sleep(2); continue
                    d = await r.json(); ch = d["choices"][0]
                    if ch.get("finish_reason") != "stop":
                        continue
                    return (ch["message"].get("content") or "").strip()
            except Exception:
                await asyncio.sleep(1)
        return ""

    t0 = time.time()
    async with aiohttp.ClientSession() as sess:
        for c0 in range(0, len(todo), CHUNK):
            chunk = todo[c0:c0 + CHUNK]
            outs = await asyncio.gather(*[call(sess, PROMPT.format(s=s[:2500])) for s in chunk])
            for s, o in zip(chunk, outs):
                done[key(s)] = o
            json.dump(done, open(OUTF + ".tmp", "w")); os.replace(OUTF + ".tmp", OUTF)
            n = min(c0 + CHUNK, len(todo))
            print(f"  {n}/{len(todo)}  {n/(time.time()-t0)*3600:.0f}/h", flush=True)
    print("done ->", OUTF)


if __name__ == "__main__":
    asyncio.run(main())
