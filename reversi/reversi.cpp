#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
using namespace std;

typedef vector < vector <int> > boardGame;

struct index {
	int x, y;
};

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

// Gives the number of opponents pieces turned if positioned here (at ind)
int getTurnOver(reversiGame gameBoard, index ind) {
	int ans = 0, i, j, tmpVal = 0;
	
	// Horizontal and vertical gains

	i = ind.x, j = ind.y;
	tmpVal = 0, j++;
	while ((j < gameBoard.col) && (gameBoard.board[i][j] != gameBoard.turn)) {
		j++;
		tmpVal++;
	}
	if (j < gameBoard.col) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, j--;
	while ((j >= 0) && (gameBoard.board[i][j] != gameBoard.turn)) {
		j--;
		tmpVal++;
	}
	if (j >= 0) {
		ans += tmpVal;
	}
	
	i = ind.x, j = ind.y;
	tmpVal = 0, i++;
	while ((i < gameBoard.row) && (gameBoard.board[i][j] != gameBoard.turn)) {
		i++;
		tmpVal++;
	}
	if (i < gameBoard.row) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, i--;
	while ((i >= 0) && (gameBoard.board[i][j] != gameBoard.turn)) {
		i--;
		tmpVal++;
	}
	if (i >= 0) {
		ans += tmpVal;
	}

	// Diagonal gains

	i = ind.x, j = ind.y;
	tmpVal = 0, j++, i++;
	while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] != gameBoard.turn)) {
		j++;
		i++;
		tmpVal++;
	}
	if ((j < gameBoard.col) && (i < gameBoard.row)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, j--, i++;
	while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] != gameBoard.turn)) {
		j--;
		i++;
		tmpVal++;
	}
	if ((j >= 0) && (i < gameBoard.row)) {
		ans += tmpVal;
	}
	
	i = ind.x, j = ind.y;
	tmpVal = 0, i--, j++;
	while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] != gameBoard.turn)) {
		i--;
		j++;
		tmpVal++;
	}
	if ((i >= 0) && (j < gameBoard.col)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, i--, j--;
	while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] != gameBoard.turn)) {
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
index getOptimalMove(reversiGame& gameBoard, boardGame risk, double tradeOff = 0.5) {
	index optInd{0, 0}, tmpInd;
	int i, j, a, b;
	double maxVal = 0, tmpVal;
	for (i = 0;i < gameBoard.candList.size(); i++) {
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

int main() {
	reversiGame gameBoard;
	initBoard(gameBoard, 10, 10);
	getBoardInput(gameBoard);
	boardGame risk;
	getRiskRegions(gameBoard, risk);
	index optMove = getOptimalMove(gameBoard, risk, 0.5);
	cout << optMove.x << " AND " << optMove.y << endl;
    return 0;
}
