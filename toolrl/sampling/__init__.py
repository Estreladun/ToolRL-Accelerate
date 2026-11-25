"""Budget-matched, no-replacement difficulty-aware rollout sampling."""
import random
from ..rewards import calls

BUCKETS = ('easy', 'medium', 'hard')


def difficulty(row):
    if 'difficulty' in row:
        return row['difficulty']
    truth = calls(row['answer'])
    return row['prompt_tokens'] + 32 * len(truth) + 8 * sum(len(c['arguments']) for c in truth)


class RolloutSampler:
    def __init__(self, rows, budget, seed, adaptive=False, alpha=.1, epsilon=.2, delta=.05, warmup=2):
        if not 1 <= budget <= len(rows):
            raise ValueError('GRPO budget must be within the candidate pool size')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('duplicate candidate IDs')
        if not 0 < alpha <= 1 or not 0 < epsilon <= 1 or delta <= 0 or warmup < 0:
            raise ValueError('invalid sampler settings')
        self.rng = random.Random(seed)
        self.budget, self.adaptive = budget, adaptive
        self.alpha, self.epsilon, self.delta, self.warmup = alpha, epsilon, delta, warmup
        self.q = dict.fromkeys(BUCKETS, .5)
        self.n = dict.fromkeys(BUCKETS, 0)
        self.remaining = {k: [] for k in BUCKETS}
        ranked = sorted(rows, key=lambda r: (difficulty(r), r['id']))
        self.bucket_of = {}
        for i, row in enumerate(ranked):
            k = BUCKETS[3*i//len(ranked)]
            self.remaining[k].append(row)
            self.bucket_of[row['id']] = k
        self.uniform = self.rng.sample(rows, budget) if not adaptive else []
        self.drawn = 0

    def probabilities(self):
        active = [k for k in BUCKETS if self.remaining[k]]
        if not active:
            return {}
        cold = [k for k in active if self.n[k] < self.warmup]
        if cold:
            # Round-robin warm-up among the least observed buckets.
            low = min(self.n[k] for k in cold)
            cold = [k for k in cold if self.n[k] == low]
            return {k: 1/len(cold) for k in cold}
        denom = sum(self.q[k]+self.delta for k in active)
        return {k: (1-self.epsilon)*(self.q[k]+self.delta)/denom + self.epsilon/len(active) for k in active}

    def draw(self):
        if self.drawn >= self.budget:
            raise StopIteration
        if self.adaptive:
            probs = self.probabilities()
            bucket = self.rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
            row = self.remaining[bucket].pop(self.rng.randrange(len(self.remaining[bucket])))
        else:
            row = self.uniform[self.drawn]
            bucket = self.bucket_of[row['id']]
            self.remaining[bucket].remove(row)
        self.drawn += 1
        return row, bucket

    def observe(self, bucket, effective):
        if effective not in (0, 1):
            raise ValueError('effective must be binary')
        self.n[bucket] += 1
        self.q[bucket] = (1-self.alpha)*self.q[bucket] + self.alpha*effective
