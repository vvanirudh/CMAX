import heapq


class Dijkstra:
    def __init__(self, successors_fn, actions):
        self.successors_fn = successors_fn
        self.actions = actions

    def get_dijkstra_heuristic(self, goal_node):
        closed_set = set()
        open = []

        g = goal_node._g = 0

        count = 0
        goal_triplet = [g, count, goal_node]
        heapq.heappush(open, goal_triplet)
        count += 1
        open_d = {goal_node: goal_triplet}

        while True:
            if len(open) == 0:
                # Expanded all states
                break

            g, _, node = heapq.heappop(open)
            del open_d[node]

            closed_set.add(node)
            for action in self.actions:
                neighbor, cost = self.successors_fn(node, action)
                if neighbor in closed_set:
                    continue

                tentative_g = node._g + cost
                if neighbor not in open_d:
                    g = neighbor._g = tentative_g
                    triplet = [g, count, neighbor]
                    heapq.heappush(open, triplet)
                    open_d[neighbor] = triplet
                    count += 1
                else:
                    neighbor = open_d[neighbor][2]
                    if tentative_g < neighbor._g:
                        neighbor._g = tentative_g
                        open_d[neighbor][0] = tentative_g
                        heapq.heapify(open)

        # Return closed list
        return closed_set
