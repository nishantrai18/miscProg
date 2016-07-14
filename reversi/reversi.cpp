#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct index {
	int x, y;
};

struct reversiGame {
	vector < vector <int> > board;
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
				index tmpInd (i, j);				
				candList.append(tmpInd);
			}
		}
	}
	scanf("%d", &gameBoard.turn);
}

int main() {
	reversiGame gameBoard;
	getBoardInput(gameBoard);
    return 0;
}
