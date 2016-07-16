#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
using namespace std;

typedef vector < vector <int> > boardGame;

struct index {
	int x, y;
};

bool operator ==(const index& x, const index& y) {
    return std::tie(x.x, x.y) == std::tie(y.x, y.y);
}

bool compareIndex(const index& a, const index& b) {
    if (a.x == b.x) {
    	return a.y < b.y;
    }
    return a.x < b.x;
}

struct reversiGame {
	boardGame board;
	// The current board status
	vector <index> candList;
	// The possible candidates for the current board
	int row, col, turn;
	// The sizes of the board
};

void initBoard(reversiGame& gameBoard, int row, int col) {
	int i;
	(gameBoard.board).resize(row);
	for (i = 0; i<row; i++) {
		(gameBoard.board[i]).resize(col);
	}
	gameBoard.row = row;
	gameBoard.col = col;
	gameBoard.turn = 0;
}

void getBoardInput(reversiGame& gameBoard) {
	int i, j;
	for (i = 0; i < gameBoard.row; i++) {
		for (j = 0; j < gameBoard.col; j++) {
			scanf("%d", &gameBoard.board[i][j]);
			if (gameBoard.board[i][j] == 3) {
				index tmpInd {i, j};				
				gameBoard.candList.push_back(tmpInd);
			}
		}
	}
	scanf("%d", &gameBoard.turn);
}

double getBoardScore(reversiGame gameBoard);
reversiGame getNewState(reversiGame gameBoard, index move);
int getTurnOver(reversiGame gameBoard, index ind);
void updateCandList(reversiGame& gameBoard);
void turnOver(reversiGame& gameBoard, index ind);
void getRiskRegions(reversiGame gameBoard, boardGame& risk);
index getOptimalMoveMethodA(reversiGame& gameBoard, boardGame risk, double tradeOff = 0.5);
index getOptimalMoveMethodB(reversiGame& gameBoard);
double minimax(reversiGame gameBoard, int ourTurn, int row, int col, int depth, index &optMove, bool abPrune = false);

int main() {
	reversiGame gameBoard;
	initBoard(gameBoard, 10, 10);
	getBoardInput(gameBoard);
	boardGame risk;
	getRiskRegions(gameBoard, risk);
	// index optMove = getOptimalMoveMethodA(gameBoard, risk, 0.5);
	index optMove = getOptimalMoveMethodB(gameBoard);
	cout << optMove.x << " AND " << optMove.y << endl;
    return 0;
}

void getCandList(reversiGame& gameBoard) {
	// Check each index for being a valid move and update the candList

	int i, j, a, b;
	vector <index> tmpCandList;
	gameBoard.candList.clear();

	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {

			if (gameBoard.board[a][b] != gameBoard.turn) {
				
				i = a, j = b;
				if ((j >= 0) && (gameBoard.board[i][j-1] == gameBoard.turn)) {
					tmpVal = 0, j++;
					while ((j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j++;
					}
					if (j < gameBoard.col) {
						tmpCandList.push_back(index {i, j});
					}
				}

				i = a, j = b;
				if ((j < (gameBoard.col - 1)) && (gameBoard.board[i][j+1] == gameBoard.turn)) {
					tmpVal = 0, j--;
					while ((j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j--;
					}
					if (j >= 0) {
						tmpCandList.push_back(index {i, j});
					}
				}

				i = a, j = b;
				if ((i >= 0) && (gameBoard.board[i-1][j] == gameBoard.turn)) {
					tmpVal = 0, i++;
					while ((i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i++;
					}
					if (i < gameBoard.row) {
						tmpCandList.push_back(index {i, j});
					}
				}

				i = a, j = b;
				if ((i < (gameBoard.row - 1)) && (gameBoard.board[i+1][j] == gameBoard.turn)) {
					tmpVal = 0, i--;
					while ((i >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i--;
					}
					if (i >= 0) {
						tmpCandList.push_back(index {i, j});
					}
				}

				// Diagonal gains

				i = a, j = b;
				if ((i >= 0) && (j >= 0) && (gameBoard.board[i-1][j-1] == gameBoard.turn)) {
					tmpVal = 0, j++, i++;
					while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j++;
						i++;
					}
					if ((j < gameBoard.col) && (i < gameBoard.row)) {
						tmpCandList.push_back(index {i, j});
					}
				}

				i = a, j = b;	
				if ((j < (gameBoard.col - 1)) && (i >= 0) && (gameBoard.board[i-1][j+1] == gameBoard.turn)) {
					tmpVal = 0, j--, i++;
					while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j--;
						i++;
					}
					if ((j >= 0) && (i < gameBoard.row)) {
						tmpCandList.push_back(index {i, j});
					}
				}

				i = a, j = b;
				if ((i < (gameBoard.row - 1)) && (j >= 0) && (gameBoard.board[i+1][j-1] == gameBoard.turn)) {
					tmpVal = 0, i--, j++;
					while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i--;
						j++;
					}
					if ((i >= 0) && (j < gameBoard.col)) {
						tmpCandList.push_back(index {i, j});
					}
				}

				i = a, j = b;
				if ((i < (gameBoard.row - 1)) && (j < (gameBoard.col - 1)) && (gameBoard.board[i+1][j+1] == gameBoard.turn)) {
					tmpVal = 0, i--, j--;
					while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i--;
						j--;
					}
					if ((i >= 0) && (j >= 0)) {
						tmpCandList.push_back(index {i, j});
					}
				}
			}
		}
	}

	sort(tmpCandList.begin(), tmpCandList.end(), compareIndex);
	for (i = 0; i < tmpCandList.size(); i++) {
		while ((i < (tmpCandList.size() - 1)) && (tmpCandList[i] == tmpCandList[i+1]))
			i++;
		gameBoard.candList.push_back(tmpCandList[i]);
	}
}

void turnOver(reversiGame& gameBoard, index ind){
	int i, j, turn = gameBoard.turn;
	
	// Horizontal and vertical gains

	i = ind.x, j = ind.y;
	j++;
	while ((j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j++;
		gameBoard.board[i][j] = turn;
	}

	i = ind.x, j = ind.y;
	j--;
	while ((j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j--;
		gameBoard.board[i][j] = turn;
	}
	
	i = ind.x, j = ind.y;
	i++;
	while ((i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i++;
		gameBoard.board[i][j] = turn;
	}
	
	i = ind.x, j = ind.y;
	i--;
	while ((i >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		gameBoard.board[i][j] = turn;
	}
	
	// Diagonal gains

	i = ind.x, j = ind.y;
	j++, i++;
	while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j++;
		i++;
		gameBoard.board[i][j] = turn;
	}

	i = ind.x, j = ind.y;
	j--, i++;
	while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j--;
		i++;
		gameBoard.board[i][j] = turn;
	}
	
	i = ind.x, j = ind.y;
	i--, j++;
	while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		j++;
		gameBoard.board[i][j] = turn;
	}

	i = ind.x, j = ind.y;
	i--, j--;
	while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		j--;
		gameBoard.board[i][j] = turn;
	}
}

double getBoardScore(reversiGame gameBoard) {
	int i, j;
	double score = 0;

	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {
			if (gameBoard.board[i][j] == gameBoard.turn)
				score += 1.0;
			else if (gameBoard.board[i][j] == (3 - gameBoard.turn))
				score -= 1.0
		}
	}

	return score;
}

reversiGame getNewState(reversiGame gameBoard, index move) {
	// Check validity of move
	// Copy original board, turn over the board pieces accoridingly
	// Change the turn, compute the candidate list for the new player

	reversiGame newGame = gameBoard;
	if (gameBoard.board[move.x][move.y] != 3) {
		cout << "ERROR: This shouldn't be printed!\n";
		return newGame;
	}

	turnOver(newGame, move);
	newGame.turn = 3 - newGame.turn;
	getCandList(newGame);

	return newGame;
}

/*
	// The minimax algorithm, traverses the tree recursively.
	// Need to further add table lookups and alpha beta pruning
	// Also add option for fooling the opponent (Can experiment)
	// (Not choosing the min if reasonable options left)
	// Also need to make another version which seeks to 
	// maximize the average values of the top k moves.

	// Input:
	// // gameRep refers to the game board representation in form of a char array for speed
	// // Probably faster than vectors of vectors. 
	// turn is either 1 or 2. It refers to our turn ID, useful for deciding min or max node.
	// gameBoard is the gameBoard being passed
	// row, col: Size of the board
	// Depth refers to the depth we need to traverse currently.
	// Output:
	// Score of the best possible result
	// index i.e. the best move
*/
double minimax(reversiGame gameBoard, int ourTurn, int row, int col, int depth, index &optMove, bool abPrune = false) {
	// Check if depth is 0, if it is, then compute the score using getScore()
	// Traverse over the possible moves and create the new game state after performing each move
	// Recursively call the function minimax for each state, store the min or max
	// Alpha beta pruning: Also pass variables alpha, beta for pruning.
	// Fooling heuristic (Probably not needed): Choose alternate moves (Only do this in intermediate stages)
	// Finally check for the minimum or maximum value and modify the optMove variable to store the best move.

	double optVal;
	// Stores the optimal value at this node

	if (depth == 0) {
		double score = getBoardScore(gameBoard);
		return score;
	}
	else {
		int i, j;
		vector <reversiGame> gameList;
		vector <double> scoreList;
		for (i = 0; i < gameBoard.candList.size(); i++) {
			reversiGame tmpGame = getNewState(gameBoard, candList[i]);
			gameList.push_back(tmpGame);
		}
		if (!abPrune) {
			for (i = 0; i < gameList.size(); i++) {
				double tmpScore = minimax(gameList[i], ourTurn, row, col, depth, optMove, abPrune);
				scoreList.push_back(tmpScore);
			}
			if (gameBoard.turn == ourTurn) {
				// This is a max node, thus we maximize our score
				optVal = 0;
				for (i = 0; i < scoreList.size(); i++) {
					if (optVal < scoreList[i]) {
						optVal = scoreList[i];
						optMove = gameBoard.candList[i];
					}
				}
			}
			else {
				// This is a min node, thus they try to minimize our score
				optVal = 100000;
				for (i = 0; i < scoreList.size(); i++) {
					if (optVal > scoreList[i]) {
						optVal = scoreList[i];
						optMove = gameBoard.candList[i];
					}
				}
			}
		}
		else {
			cout << "Not implemented alpha beta pruning!\n";
		}
	}

	return optVal;
}

// Gives the number of opponents pieces turned if positioned here (at ind)
int getTurnOver(reversiGame gameBoard, index ind) {
	int ans = 0, i, j, tmpVal = 0;
	
	// Horizontal and vertical gains

	i = ind.x, j = ind.y;
	tmpVal = 0, j++;
	while ((j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j++;
		tmpVal++;
	}
	if (j < gameBoard.col) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, j--;
	while ((j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j--;
		tmpVal++;
	}
	if (j >= 0) {
		ans += tmpVal;
	}
	
	i = ind.x, j = ind.y;
	tmpVal = 0, i++;
	while ((i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i++;
		tmpVal++;
	}
	if (i < gameBoard.row) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, i--;
	while ((i >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		tmpVal++;
	}
	if (i >= 0) {
		ans += tmpVal;
	}

	// Diagonal gains

	i = ind.x, j = ind.y;
	tmpVal = 0, j++, i++;
	while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j++;
		i++;
		tmpVal++;
	}
	if ((j < gameBoard.col) && (i < gameBoard.row)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, j--, i++;
	while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j--;
		i++;
		tmpVal++;
	}
	if ((j >= 0) && (i < gameBoard.row)) {
		ans += tmpVal;
	}
	
	i = ind.x, j = ind.y;
	tmpVal = 0, i--, j++;
	while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		j++;
		tmpVal++;
	}
	if ((i >= 0) && (j < gameBoard.col)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, i--, j--;
	while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		j--;
		tmpVal++;
	}
	if ((i >= 0) && (j >= 0)) {
		ans += tmpVal;
	}

	return ans;
}

// Gives the risk regions and the associated score
void getRiskRegions(reversiGame gameBoard, boardGame& risk) {
	int i, j;
	risk.resize(gameBoard.row);
	for (i = 0 ; i < gameBoard.row; i++) {
		risk[i].resize(gameBoard.col);
	}
	for (i = 0; i < gameBoard.row; i++) {
		for (j = 0; j < gameBoard.col; j++) {
			if ((j == 0) || (j == gameBoard.col - 1)) {
				if ((i == 0) || (i == gameBoard.row - 1)) {
					risk[i][j] = 10;
				}
				else if ((i == 1) || (i == gameBoard.row - 2)) {
					risk[i][j] = 0;
				}
				else {
					risk[i][j] = 7;
				}
			} 
			else if ((j == 1) || (j == gameBoard.col - 2)) {
				if ((i < 2) || (i > gameBoard.row - 3)) {
					risk[i][j] = 0;
				}
				else {
					risk[i][j] = 2;
				}
			} 
			else {
				if ((i == 0) || (i == gameBoard.row - 1)) {
					risk[i][j] = 7;
				}
				else if ((i == 1) || (i == gameBoard.row - 2)) {
					risk[i][j] = 2;
				}
				else {
					risk[i][j] = 5;
				}
			}
		}
	}
}

// Gives the optimal move for the current board status
// Method A : Greedy Strategy using a weighted sum of turnOvers and riskRegions
index getOptimalMoveMethodA(reversiGame& gameBoard, boardGame risk, double tradeOff = 0.5) {
	index optInd{0, 0}, tmpInd;
	int i, j, a, b;
	double maxVal = 0, tmpVal;
	for (i = 0; i < gameBoard.candList.size(); i++) {
		tmpInd = gameBoard.candList[i];
		a = getTurnOver(gameBoard, tmpInd);
		b = risk[gameBoard.candList[i].x][gameBoard.candList[i].y];
		tmpVal = (a*(1.0) + (tradeOff*b));
		if (tmpVal > maxVal) {
			maxVal = tmpVal;
			optInd = tmpInd;
		}
	}
	return optInd;
}

// Gives the optimal move for the current board status
// Method B : Minmax tree search
// Input : Game board, Depth limit
index getOptimalMoveMethodB(reversiGame& gameBoard) {
	index optInd{0, 0}, tmpInd;
	minimax(gameBoard, gameBoard.turn, gameBoard.row, gameBoard.col, 6, optInd);
	return optInd;
}
