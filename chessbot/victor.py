import random as rnd
import timeit
from math import sqrt, log

from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from .bot import ChessBot

class ChessBotVictor(ChessBot):
    def __init__(self, name, opt_dict = None):
        super().__init__(name, opt_dict)
        self.depth = opt_dict['depth']        
        self.is_white = True

        self.random_table = [[rnd.randint(0, 2**64) for _ in range(12)] for _ in range(64)]
        self.transposition = dict()

        self.pawns_eval_white = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]

        self.pawns_eval_black = [i for i in reversed(self.pawns_eval_white)]

        self.knights_eval_white = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]

        self.knights_eval_black = [i for i in reversed(self.knights_eval_white)]

        self.bishops_eval_white = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]

        self.bishops_eval_black = [i for i in reversed(self.bishops_eval_white)]

        self.rooks_eval_white = [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             0,  0,  0,  5,  5,  0,  0,  0
        ]

        self.rooks_eval_black = [i for i in reversed(self.rooks_eval_white)]

        self.queen_eval_white = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
             -5,  0,  5,  5,  5,  5,  0, -5,
              0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.queen_eval_black = [i for i in reversed(self.queen_eval_white)]

        self.king_eval_white = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
             20, 20,  0,  0,  0,  0, 20, 20,
             20, 30, 10,  0,  0, 10, 30, 20
        ]

        self.king_eval_black = [i for i in reversed(self.king_eval_white)]

        self.king_end_game_eval_white = [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ]

        self.king_end_game_eval_black = [i for i in reversed(self.king_end_game_eval_white)]

    def is_end_game(self, board):
        n_pawns = len(board.pieces(PAWN, True)) + len(board.pieces(PAWN, False))
        n_knights = len(board.pieces(KNIGHT, True)) + len(board.pieces(KNIGHT, False))
        n_bishops = len(board.pieces(BISHOP, True)) + len(board.pieces(BISHOP, False))
        n_rooks = len(board.pieces(ROOK, True)) + len(board.pieces(ROOK, False))
        n_queens = len(board.pieces(QUEEN, True)) + len(board.pieces(QUEEN, False))
        n_kings = len(board.pieces(KING, True)) + len(board.pieces(KING, False))

        if n_pawns + n_knights + n_bishops + n_rooks + n_queens + n_kings <= 22:
            return True
        else:
            return False

    def calc_heuristic_score(self, board):
        if board.is_game_over():
            result = board.result()
            if (self.is_white and result == '1-0') or ((not self.is_white) and result == '0-1'):
                return 10**5
            elif (self.is_white and result == '0-1') or ((not self.is_white) and result == '1-0'):
                return -10**5
            else:
                return 0

        res = 0
        scores = {PAWN:100, KNIGHT:300, BISHOP:300, ROOK:500, QUEEN: 900, KING:9000}
        color_factors = {self.is_white: 1, (not self.is_white): -1}

        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                res += scores[piece.piece_type] * color_factors[piece.color]

                if piece.color == self.is_white:
                    if piece.piece_type == PAWN:
                        if self.is_white:
                            res += self.pawns_eval_white[square]
                        else:
                            res += self.pawns_eval_black[square]

                    elif piece.piece_type == KNIGHT:
                        if self.is_white:
                            res += self.knights_eval_white[square]
                        else:
                            res += self.knights_eval_black[square]

                    elif piece.piece_type == BISHOP:
                        if self.is_white:
                            res += self.bishops_eval_white[square]
                        else:
                            res += self.bishops_eval_black[square]

                    elif piece.piece_type == ROOK:
                        if self.is_white:
                            res += self.rooks_eval_white[square]
                        else:
                            res += self.rooks_eval_black[square]

                    elif piece.piece_type == QUEEN:
                        if self.is_white:
                            res += self.queen_eval_white[square]
                        else:
                            res += self.queen_eval_black[square]

                    elif piece.piece_type == KING and self.is_end_game(board):
                        if self.is_white:
                            res += self.king_end_game_eval_white[square]
                        else:
                            res += self.king_end_game_eval_black[square]

                    else:
                        if self.is_white:
                            res += self.king_eval_white[square]
                        else:
                            res += self.king_eval_black[square]

        self.transposition[self.zobrist_hash(board)] = res
        return res

    def get_moves_to_explore(self, board):
        moves_to_explore = []
        moves = self.possible_moves(board)

        if False:
            scores = []
            for move in moves:
                board_copy = board.copy()
                board_copy.push(move)

                scores.append(self.calc_heuristic_score(board_copy))

            moves_dict = dict(list(zip(scores, moves)))
            scores.sort(reverse=True)

            for i in range(4):
                key = scores[i]
                moves_to_explore.append(moves_dict[key])

            for i in range(2):
                moves_to_explore.append(rnd.choice(moves))

            for i in range(4):
                key = scores[len(scores)-1-i]
                moves_to_explore.append(moves_dict[key])

            return moves_to_explore

        else:
            return moves

    def zobrist_hash(self, board):
        _hash = 0
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                j = piece.piece_type
                if not piece.color:
                    j += 6
                _hash ^= self.random_table[i][j-1]

        return _hash

    def minimax(self, board, depth, alpha, beta):
        if self.zobrist_hash(board) in self.transposition.keys():
            return self.transposition[self.zobrist_hash(board)]

        if depth == 1 or board.is_game_over():
            return self.calc_heuristic_score(board)

        elif board.turn == self.is_white:
            v = -10**6

            for a in self.get_moves_to_explore(board):
                board_copy = board.copy()
                board_copy.push(a)

                v = max(v, self.minimax(board_copy, depth-1, alpha, beta))
                alpha = max(alpha, v)

                if alpha >= beta:
                    break

            return v

        else:
            v = 10**6

            for a in self.get_moves_to_explore(board):
                board_copy = board.copy()
                board_copy.push(a)
         
                v = min(v, self.minimax(board_copy, depth-1, alpha, beta))
                beta = min(beta, v)

                if alpha >= beta:                  
                    break

            return v

    def move(self, board):
        self.is_white = board.turn
        best_score = -10**6
        current_score = 0
        best_move = None

        for move in self.get_moves_to_explore(board):
            board_copy = board.copy()
            board_copy.push(move)
            current_score = self.minimax(board_copy, self.depth, -10**6, 10**6)

            if current_score > best_score:
                best_score = current_score
                best_move = move

        return best_move


# # # # # # # # # # # # # # # # # # # # # #
# Chess Bot using Monte Carlo Tree Search #
# # # # # # # # # # # # # # # # # # # # # #

class ChessBotMonteCarlo(ChessBot):
    def __init__(self, name, opt_dict = None):
        super().__init__(name, opt_dict)
        self.n_simulations = opt_dict['n_simulations']
        self.n_interations = opt_dict['n_iterations']
        self.is_white = True

    class Node:
        def __init__(self, n_wins, n_simulations, board, parent, children):
            self.n_wins = n_wins
            self.n_simulations = n_simulations
            self.board = board
            self.parent = parent
            self.children = children

        def add_child(self, node):
            node.parent = self
            self.children.append(node)

    def get_uct(self, node, c=2):
        return (node.n_wins / node.n_simulations) + sqrt(c*log(node.parent.n_simulations)/node.n_simulations)

    def pick_best_child(self, root):
        best_uct = -10**6
        current_uct = 0
        best_child = None

        for child in root.children:
            current_uct = self.get_uct(child)
            if current_uct > best_uct:
                best_uct = current_uct
                best_child = child

        return best_child

    def simulate(self, node):
        n_wins = 0
  
        for _ in range(self.n_simulations):
            board_copy = node.board.copy()

            # print('Random Playout start...')
            while board_copy.result() == '*':
                moves = self.possible_moves(board_copy)
                board_copy.push(moves[rnd.randint(0, len(moves)-1)])
            # print('Random Playout ended.')            

            if (board_copy.result() == '1-0' and self.is_white) or (board_copy.result() == '0-1' and not self.is_white):
                n_wins += 1
            else:
                pass

        return self.Node(n_wins, self.n_simulations, node.board.copy(), None, [])

    def backpropagate(self, node):
        for child in node.children:
            node.n_wins += child.n_wins
            node.n_simulations += child.n_simulations

        while node.parent != None:
            node.parent.n_wins += node.n_wins
            node.parent.n_simulations += node.n_simulations

            node = node.parent

        return node

    def monte_carlo_tree_search(self, root):
        for _ in range(self.n_interations):
            # print('Wins: ' + str(root.n_wins))
            # print('Simulations: ' + str(root.n_simulations))            

            # Selection
            node = root
            while len(node.children) > 0:
                node = self.pick_best_child(node)

            if len(self.possible_moves(node.board)) == 0:
                break

            # Expansion
            expanded_nodes = []
            moves = self.possible_moves(node.board)
            for _ in range(len(moves)//2):
                board_copy = node.board.copy()            
                board_copy.push(moves[rnd.randint(0, len(moves)-1)])
                expanded_nodes.append(self.Node(0, 0, board_copy, None, []))
            
            # Simulation
            for expanded_node in expanded_nodes:
                node.add_child(self.simulate(expanded_node))

            # Backpropagation
            root = self.backpropagate(node)          

        return root

    def move(self, board):
        self.is_white = board.turn
        root = self.monte_carlo_tree_search(self.Node(0, 0, board, None, []))
        
        best_stat = -10^6
        current_state = 0
        best_node = None

        for node in root.children:
            current_state = node.n_wins / node.n_simulations
            if current_state > best_stat:
                best_stat = current_state
                best_node = node

        return best_node.board.pop()
