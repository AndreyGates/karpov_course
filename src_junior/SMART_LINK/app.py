"""SMART-LINK (pt.1)"""
from collections import defaultdict
import numpy as np
import uvicorn
from fastapi import FastAPI

# Global stats saving

# offer ID - conversions link
offer_actions = defaultdict(int)
# click ID - order ID link
pending_clicks = defaultdict(int)
# order ID - reward link
offer_rewards = defaultdict(float)
# offer ID - No. of clicks link
offer_clicks = defaultdict(int)

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Reset stats"""
    offer_actions.clear()
    pending_clicks.clear()
    offer_rewards.clear()
    offer_clicks.clear()

@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)

    # if click_id is not among the pending clicks
    if click_id not in pending_clicks:
        return None

    # otherwise, update the corresponding offer stats
    # define the conversion
    is_conversion = reward > 0
    # save the offer ID to update its stats
    offer_id = pending_clicks[click_id]
    # deleting the click ID since we save the offer ID to update the stats
    del pending_clicks[click_id]
    # update the stats
    if reward > 0:
        offer_actions[offer_id] += 1
        offer_rewards[offer_id] += reward

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward
    }

    return response

@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    clicks = offer_clicks[offer_id] # No. of clicks
    conversions = offer_actions[offer_id] # No. of actions
    reward = offer_rewards[offer_id] # offer reward
    cr = None # conversion rate
    rpc = None # revenue per click

    # checking for zero stats
    if clicks == 0:
        cr, rpc = 0.0, 0.0
    else:
        cr = conversions / clicks
        rpc = reward / clicks
    response = {
        "offer_id": offer_id,
        "clicks": clicks,
        "conversions": conversions,
        "reward": reward,
        "cr": cr,
        "rpc": rpc
    }
    return response

@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Greedy sampling"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    ### Sample top offer ID
    top_offer, top_rpc = None, None
    # defining (at the end) the mode - random or greedy
    is_greedy = True
    # if there are fewer than 100 clicks, pick a random offer:
    if sum(offer_clicks.values()) < 100:
        top_offer = np.random.choice(offers_ids)
        is_greedy = False
    # otherwise, go greedy maximing RPC
    else:
        # in case all rewards are zero, the first offer stands
        if sum(offer_rewards.values()) == 0:
            top_offer = offer_ids[0]

        else:
            # if the offers doesn't have clicks, set rpc = 0
            if top_offer not in offer_clicks:
                reward, clicks = 0.0, 0
            else:
                reward, clicks = offer_rewards[top_offer], offer_clicks[top_offer]
                top_rpc = reward / clicks # revenue per click

            # trying to find the offer maximizing RPC
            for off_id in offers_ids:
                # if the offer doesn't have clicks, skip it
                if off_id not in offer_clicks:
                    continue
                reward, clicks = offer_rewards[off_id], offer_clicks[off_id]
                if reward / clicks > top_rpc:
                    top_offer = off_id
                    top_rpc = reward / clicks

    # update info
    pending_clicks[click_id] = top_offer
    offer_clicks[top_offer] = offer_clicks[top_offer] + 1

    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": top_offer,
        "sampler": "greedy" if is_greedy else "random"
    }

    return response

def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")

if __name__ == "__main__":
    main()
