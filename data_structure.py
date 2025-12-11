import heapq


class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False
        
    def __gt__(self, other):
        return self.prob > other.prob
    
    def __ge__(self, other):
        return self.prob >= other.prob
    
    def __lt__(self, other):
        return self.prob < other.prob
    
    def __le__(self, other):
        return self.prob <= other.prob
    
    def __eq__(self, other):
        return self.prob == other.prob
    
    def __ne__(self, other):
        return self.prob != other.prob
    
    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")


class PriorityQueue():
    def __init__(self):
        self.queue = []

    def put(self, obj):
        # FIX: Remove the negative sign.
        # We want a Min-Heap (pop the lowest score/cost first)
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def empty(self):
        return len(self.queue) == 0
    