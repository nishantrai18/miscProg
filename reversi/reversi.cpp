#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <map>
using namespace std;

typedef vector < vector <int> > boardGame;

struct index {
	int x, y;
};

bool operator ==(const index& x, const index& y) {
    return ((x.x == y.x) && (x.y == y.y));
}

bool operator !=(const index& x, const index& y) {
    return !((x.x == y.x) && (x.y == y.y));
}

bool compareIndex(const index& a, const index& b) {
    if (a.x == b.x) {
    	return a.y < b.y;
    }
    return a.x < b.x;
}

index createIndex(int a, int b) {
	index retVal {a, b};
	return retVal;
}

map<string, double> stateMap;

int totCount = 0;
double currVal = 0;

struct reversiGame {
	boardGame board;
	// The current board status
	vector <index> candList;
	// The possible candidates for the current board
	int row, col, turn;
	// The sizes of the board
	int cntUs, cntOpp;
	// Counts of the pieces
	bool end;
	// State of the board, whether the game has ended or not.
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
	gameBoard.end = false;
	gameBoard.cntUs = 0;
	gameBoard.cntOpp = 0;
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
	
	int a, b;
	
	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {
			if (gameBoard.board[a][b] == gameBoard.turn) {
				gameBoard.cntUs++;
			}
			else if (gameBoard.board[a][b] == (3 - gameBoard.turn)) {
				gameBoard.cntOpp++;
			}
		}
	}
}

string getBoardString(reversiGame gameBoard) {
	string retVal = "";
	int i, j, a, b;
	
	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {
			retVal += '0' + gameBoard.board[a][b];
		}
	}

	retVal += '0' + gameBoard.turn;
	return retVal;
}

double getStoredState(reversiGame gameBoard) {
	string key = getBoardString(gameBoard);
	if (stateMap.find(key) != stateMap.end() ) {
		// cout << "I WAS HELPFUL!\n";
		return stateMap[key];
	}
	else {
		return -1000000;
	}
}

double getBoardScore(reversiGame gameBoard);
void printGameState(reversiGame gameBoard, boardGame risk, int ourTurn);
reversiGame getNewState(reversiGame gameBoard, boardGame risk, index move);
int getTurnOver(reversiGame gameBoard, index ind, int turn);
void updateCandList(reversiGame& gameBoard);
void turnOver(reversiGame& gameBoard, index ind);
void getRiskRegions(reversiGame gameBoard, boardGame& risk);
index getOptimalMoveMethodA(reversiGame& gameBoard, double tradeOff = 0.5);
index getOptimalMoveMethodB(reversiGame& gameBoard);
int getStabilityMat(reversiGame gameBoard, boardGame& stableStat, int turn);
double minimax(reversiGame gameBoard, boardGame risk, int ourTurn, int row, int col, int depth, \
				index &optMove, double alpha, double beta, int flag = 0, bool avgPath = false, bool mapFlag = false, bool minPrune = false);
double getDynamicRisk(reversiGame gameBoard, boardGame risk, int x, int y);

int main() {
	reversiGame gameBoard;
	initBoard(gameBoard, 10, 10);
	getBoardInput(gameBoard);
	// index optMove = getOptimalMoveMethodA(gameBoard, 0.5);
	index optMove = getOptimalMoveMethodB(gameBoard);
	cout << optMove.x << " " << optMove.y << endl;
    return 0;
}

double getDynamicRisk(reversiGame gameBoard, boardGame risk, int x, int y) {
	int i = x, j = y;
	
	if ((i - j) == (-(gameBoard.col - 2))) {
		return ((gameBoard.board[0][gameBoard.col - 1] != 0) ? 0 : risk[i][j]);
	}
	else if ((i - j) == (gameBoard.row - 2)) {
		return ((gameBoard.board[gameBoard.row - 1][0] != 0) ? 0 : risk[i][j]);
	}
	else if ((i + j) == 1) {
		return ((gameBoard.board[0][0] != 0) ? 0 : risk[i][j]);
	}
	else if ((i + j) == (gameBoard.row + gameBoard.col - 3)) {
		return ((gameBoard.board[gameBoard.row - 1][gameBoard.col - 1] != 0) ? 0 : risk[i][j]);
	}
	else {
		return risk[i][j];
	}
}


double getBoardScore(reversiGame gameBoard, boardGame risk, int ourTurn) {
	int i, j, a, b;
	double score = 0, turnUs = 0, turnOpp = 0, maxTurnUs = 0, maxTurnOpp = 0;
	int cntUs = 0, cntOpp = 0;
	double cntTurnUs = 0, cntTurnOpp = 0;

	// printGameState(gameBoard, risk, ourTurn);

	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {
			if (gameBoard.board[a][b] == ourTurn) {
				cntUs++;
			}
			else if (gameBoard.board[a][b] == (3 - ourTurn)) {
				cntOpp++;
			}
			else {
				// cout <<"For index " << a << ", " << b;
				double tmpVal = getTurnOver(gameBoard, createIndex(a, b), ourTurn);
				if (tmpVal > 0.01) cntTurnUs++;
				// cout << " : Our turnOver, " << tmpVal;
				turnUs += tmpVal;
				maxTurnUs = max(maxTurnUs, tmpVal);
				tmpVal = getTurnOver(gameBoard, createIndex(a, b), 3 - ourTurn);
				// cout << ", Opp turnover, " << tmpVal << endl;
				if (tmpVal > 0.01) cntTurnOpp++;
				turnOpp += tmpVal;
				maxTurnOpp = max(maxTurnOpp, tmpVal);
			}
		}
	}

	int fact = 1, factB = 1;
	if ((cntUs + cntOpp) < 15)
		fact = 1, factB = 3;
	else if ((cntUs + cntOpp) < 40)
		fact = 2, factB = 3;
	else if ((cntUs + cntOpp) < 70)
		fact = 5, factB = 2;
	else
		fact = 10, factB = 2;

	boardGame stableStat;
	int unstableUs = getStabilityMat(gameBoard, stableStat, 3 - ourTurn);
	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {
			if (gameBoard.board[a][b] == ourTurn) {
				if (stableStat[a][b] == 0)
					score += factB * 0.25;
				else
					score -= factB * 0.125;
			}
			else if (gameBoard.board[a][b] == (3 - ourTurn)) {
				if (stableStat[a][b] > 0)
					score += factB * 0.25;
				else
					score -= factB * 0.125;
			}
		}
	}

	int unstableOpp = getStabilityMat(gameBoard, stableStat, ourTurn);

	// cout << unstableUs << " and " << unstableOpp << endl;
	// score -= (unstableUs*(0.5));
	// score += (unstableOpp*(0.4));

	// cout << "Score 0 here is " << score << endl;

	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {
			if (gameBoard.board[a][b] == ourTurn) {
				score += ((fact<6)?0.5:2.0);
				score += ((0.05) * fact * (getDynamicRisk(gameBoard, risk, a, b)));
			}
			else if (gameBoard.board[a][b] == (3 - ourTurn)) {
				score -= ((fact<6)?0.5:2.0);
				score -= ((0.05) * fact * (getDynamicRisk(gameBoard, risk, a, b)));
			}
		}
	}

	// cout << "Score 1 here is " << score << endl;

	for (i = 0; i < gameBoard.candList.size(); i++) {
		if (gameBoard.turn == ourTurn) {
			score += ((0.025) * fact * getDynamicRisk(gameBoard, risk, gameBoard.candList[i].x, gameBoard.candList[i].y));
		}
		else {
			score -= ((0.025) * fact * getDynamicRisk(gameBoard, risk, gameBoard.candList[i].x, gameBoard.candList[i].y));
		}		
	}

	// cout << "Score 2 here is " << score << endl;

	if ((cntUs + cntOpp) == 100) {
		score += (cntUs - cntOpp) * 1000.0;
	}

	if ((cntUs + cntOpp) < 90) {
		if (cntTurnUs < 0.001)
			turnUs = 1, cntTurnUs = -0.01;
		if (cntTurnOpp < 0.001)
			turnOpp = 1, cntTurnOpp = 0.01;
	}
	else {
		if (cntTurnUs < 0.001)
			turnUs = 0, cntTurnUs = -0.01;
		if (cntTurnOpp < 0.001)
			turnOpp = 0, cntTurnOpp = 0.01;		
	}

	// cout << "(turnUs) : " << ((0.1 * (turnUs))) << endl;
	// cout << "(turnOpp) : " << ((0.1 * (turnOpp))) << endl;
	// cout << "(turnUs/cntTurnUs) : " <<	((2.0 * (turnUs/cntTurnUs))) << endl;
	// cout << "(turnOpp/cntTurnOpp) : " << ((1.5 * (turnOpp/cntTurnOpp))) << endl;
	// cout << "(cntTurnUs) : " <<	(0.2*cntTurnUs) << endl;
	// cout << "(cntTurnOpp) : " << (0.2*cntTurnOpp) << endl;
	// cout << "(maxTurnUs) : " <<	(0.2*maxTurnUs) << endl;
	// cout << "(maxTurnOpp) : " << (0.2*maxTurnOpp) << endl;

	// printGameState(gameBoard, risk, ourTurn);

	score += ((0.025 * (turnUs)) / (fact*(1.0)));
	score -= ((0.025 * (turnOpp)) / (fact*(1.0)));
	score += ((0.2 * (turnUs/cntTurnUs)));
	score -= ((0.2 * (turnOpp/cntTurnOpp)));
	score += (0.5 * cntTurnUs * factB);
	score -= (0.5 * cntTurnOpp * factB);
	score += (0.4 * maxTurnUs);
	score -= (0.4 * maxTurnOpp);

	// cout << "Score 3 here is " << score << endl;

	// cout << "CONTINUE? : ";
	// int ta;
	// cin >> ta;

	return score;
}

void printGameState(reversiGame gameBoard, boardGame risk, int ourTurn) {
	int i, j;
	cout << "The game board is\n";
	for (i = 0; i < gameBoard.row; i++) {
		for (j = 0; j < gameBoard.col; j++) {
			cout << gameBoard.board[i][j];
		}
		cout << endl;
	}
	cout << "The turn is of player " << gameBoard.turn << endl;
	// cout << "The score of the board is " << getBoardScore(gameBoard, risk, ourTurn) << endl;
}

void getCandList(reversiGame& gameBoard) {
	// Check each index for being a valid move and update the candList

	int i, j, a, b;
	vector <index> tmpCandList;
	gameBoard.candList.clear();

	for (i = 0; i < gameBoard.row; i++) {
		for (j = 0; j < gameBoard.col; j++) {
			if (gameBoard.board[i][j] == 3) {
				gameBoard.board[i][j] = 0;
			}
		}
	}

	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {

			if (gameBoard.board[a][b] == (3 - gameBoard.turn)) {
				
				i = a, j = b;
				if ((j > 0) && (gameBoard.board[i][j-1] == gameBoard.turn)) {
					j++;
					while ((j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j++;
					}
					if ((j < gameBoard.col) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
					}
				}

				i = a, j = b;
				if ((j < (gameBoard.col - 1)) && (gameBoard.board[i][j+1] == gameBoard.turn)) {
					j--;
					while ((j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j--;
					}
					if ((j >= 0) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
					}
				}

				i = a, j = b;
				if ((i > 0) && (gameBoard.board[i-1][j] == gameBoard.turn)) {
					i++;
					while ((i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i++;
					}
					if ((i < gameBoard.row) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
					}
				}

				i = a, j = b;
				if ((i < (gameBoard.row - 1)) && (gameBoard.board[i+1][j] == gameBoard.turn)) {
					i--;
					while ((i >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i--;
					}
					if ((i >= 0) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
					}
				}

				// Diagonal gains

				i = a, j = b;
				if ((i > 0) && (j > 0) && (gameBoard.board[i-1][j-1] == gameBoard.turn)) {
					j++, i++;
					while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j++;
						i++;
					}
					if ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
					}
				}

				i = a, j = b;	
				if ((j < (gameBoard.col - 1)) && (i > 0) && (gameBoard.board[i-1][j+1] == gameBoard.turn)) {
					j--, i++;
					while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						j--;
						i++;
					}
					if ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
					}
				}

				i = a, j = b;
				if ((i < (gameBoard.row - 1)) && (j > 0) && (gameBoard.board[i+1][j-1] == gameBoard.turn)) {
					i--, j++;
					while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i--;
						j++;
					}
					if ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
					}
				}

				i = a, j = b;
				if ((i < (gameBoard.row - 1)) && (j < (gameBoard.col - 1)) && (gameBoard.board[i+1][j+1] == gameBoard.turn)) {
					i--, j--;
					while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
						i--;
						j--;
					}
					if ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == 0)) {
						tmpCandList.push_back(createIndex(i, j));
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
		gameBoard.board[tmpCandList[i].x][tmpCandList[i].y] = 3;
	}
}

void turnOver(reversiGame& gameBoard, index ind){
	int i, j, turn = gameBoard.turn;
	
	// Horizontal and vertical gains

	// cout << "WAS HERE\n";

	i = ind.x, j = ind.y;
	gameBoard.board[i][j] = turn;

	i = ind.x, j = ind.y;
	j++;
	while ((j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j++;
	}
	if ((j < gameBoard.col) && (gameBoard.board[i][j] == gameBoard.turn)) {
		j--;
		while (j != ind.y) {
			gameBoard.board[i][j] = turn;
			j--;
		}
	}

	i = ind.x, j = ind.y;
	j--;
	while ((j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j--;
	}
	if ((j >= 0) && (gameBoard.board[i][j] == gameBoard.turn)) {
		j++;
		while (j != ind.y) {
			gameBoard.board[i][j] = turn;
			j++;
		}
	}
	
	i = ind.x, j = ind.y;
	i++;
	while ((i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i++;
	}
	if ((i < gameBoard.row) && (gameBoard.board[i][j] == gameBoard.turn)) {
		i--;
		while (i != ind.x) {
			gameBoard.board[i][j] = turn;
			i--;
		}
	}
	
	i = ind.x, j = ind.y;
	i--;
	while ((i >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
	}
	if ((i >= 0) && (gameBoard.board[i][j] == gameBoard.turn)) {
		i++;
		while (i != ind.x) {
			gameBoard.board[i][j] = turn;
			i++;
		}
	}	
	// Diagonal gains

	i = ind.x, j = ind.y;
	j++, i++;
	while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j++;
		i++;
	}
	if ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == gameBoard.turn)) {
		j--;
		i--;
		while (j != ind.y) {
			gameBoard.board[i][j] = turn;
			i--;
			j--;
		}
	}

	i = ind.x, j = ind.y;
	j--, i++;
	while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		j--;
		i++;
	}
	if ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == gameBoard.turn)) {
		j++;
		i--;
		while (j != ind.y) {
			gameBoard.board[i][j] = turn;
			i--;
			j++;
		}
	}
	
	i = ind.x, j = ind.y;
	i--, j++;
	while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		j++;
	}
	if ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == gameBoard.turn)) {
		i++;
		j--;
		while (j != ind.y) {
			gameBoard.board[i][j] = turn;
			i++;
			j--;
		}
	}

	i = ind.x, j = ind.y;
	i--, j--;
	while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == (3 - gameBoard.turn))) {
		i--;
		j--;
	}
	if ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == gameBoard.turn)) {
		i++;
		j++;
		while (j != ind.y) {
			gameBoard.board[i][j] = turn;
			i++;
			j++;
		}
	}

	// cout << "FINISHED HERE\n";
}

// Returns number of unstable nodes
int getStabilityMat(reversiGame gameBoard, boardGame& stableStat, int turn) {
	int i, j, a, b;

	stableStat.resize(gameBoard.row);

	for (i = 0; i < gameBoard.row; i++)
		stableStat[i].resize(gameBoard.col);

	for (a = 0; a < gameBoard.row; a++)
		for (b = 0; b < gameBoard.col; b++)
			stableStat[a][b] = 0;

	for (a = 0; a < gameBoard.row; a++) {
		for (b = 0; b < gameBoard.col; b++) {
			if ((gameBoard.board[a][b] == 0) || (gameBoard.board[a][b] == 3)) {
				index ind {a, b};

				i = ind.x, j = ind.y;
				j++;
				while ((j < gameBoard.col) && (gameBoard.board[i][j] == (3 - turn))) {
					j++;
				}
				if ((j < gameBoard.col) && (gameBoard.board[i][j] == turn)) {
					j--;
					while (j != ind.y) {
						stableStat[i][j]++;
						j--;
					}
				}

				i = ind.x, j = ind.y;
				j--;
				while ((j >= 0) && (gameBoard.board[i][j] == (3 - turn))) {
					j--;
				}
				if ((j >= 0) && (gameBoard.board[i][j] == turn)) {
					j++;
					while (j != ind.y) {
						stableStat[i][j]++;
						j++;
					}
				}
				
				i = ind.x, j = ind.y;
				i++;
				while ((i < gameBoard.row) && (gameBoard.board[i][j] == (3 - turn))) {
					i++;
				}
				if ((i < gameBoard.row) && (gameBoard.board[i][j] == turn)) {
					i--;
					while (i != ind.x) {
						stableStat[i][j]++;
						i--;
					}
				}
				
				i = ind.x, j = ind.y;
				i--;
				while ((i >= 0) && (gameBoard.board[i][j] == (3 - turn))) {
					i--;
				}
				if ((i >= 0) && (gameBoard.board[i][j] == turn)) {
					i++;
					while (i != ind.x) {
						stableStat[i][j]++;
						i++;
					}
				}	
				// Diagonal gains

				i = ind.x, j = ind.y;
				j++, i++;
				while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - turn))) {
					j++;
					i++;
				}
				if ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == turn)) {
					j--;
					i--;
					while (j != ind.y) {
						stableStat[i][j]++;
						i--;
						j--;
					}
				}

				i = ind.x, j = ind.y;
				j--, i++;
				while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - turn))) {
					j--;
					i++;
				}
				if ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == turn)) {
					j++;
					i--;
					while (j != ind.y) {
						stableStat[i][j]++;
						i--;
						j++;
					}
				}
				
				i = ind.x, j = ind.y;
				i--, j++;
				while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == (3 - turn))) {
					i--;
					j++;
				}
				if ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == turn)) {
					i++;
					j--;
					while (j != ind.y) {
						stableStat[i][j]++;
						i++;
						j--;
					}
				}

				i = ind.x, j = ind.y;
				i--, j--;
				while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == (3 - turn))) {
					i--;
					j--;
				}
				if ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == turn)) {
					i++;
					j++;
					while (j != ind.y) {
						stableStat[i][j]++;
						i++;
						j++;
					}
				}
			}
		}
	}

	int ans = 0;
	for (a = 0; a < gameBoard.row; a++)
		for (b = 0; b < gameBoard.col; b++) {
			if (stableStat[a][b] > 1)
				ans += 2;
			else if (stableStat[a][b] == 1)
				ans++;
		}

	// cout << "The stability matrix is\n";
	// for (i = 0; i < gameBoard.row; i++) {
	// 	for (j = 0; j < gameBoard.col; j++) {
	// 		cout << stableStat[i][j];
	// 	}
	// 	cout << endl;
	// }

	return ans;
}

reversiGame getNewState(reversiGame gameBoard, index move, int noMoveFlag = 0) {
	// Check validity of move
	// Copy original board, turn over the board pieces accoridingly
	// Change the turn, compute the candidate list for the new player

	reversiGame newGame = gameBoard;

	if (noMoveFlag) {
		newGame.turn = 3 - newGame.turn;
		getCandList(newGame);		
	}


	if (gameBoard.board[move.x][move.y] != 3) {
		cout << "ERROR: This shouldn't be printed!\n";
		return newGame;
	}

	// cout << "Player " << gameBoard.turn << " is moving " << move.x << ", " << move.y << endl;

	turnOver(newGame, move);
	newGame.turn = 3 - newGame.turn;
	// cout << "BEFORE!\n";
	// printGameState(newGame, risk, ourTurn);
	getCandList(newGame);
	// cout << "AFTER!\n";
	// printGameState(newGame, risk, ourTurn);
	
	// cout << "The candList\n";
	// for (int i = 0; i < newGame.candList.size(); i++) {
	// 	cout << newGame.candList[i].x << " with " << newGame.candList[i].y << endl;
	// }

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
double minimax(reversiGame gameBoard, boardGame risk, int ourTurn, int row, int col, int depth,\
				index &optMove, double alpha, double beta, int flag, bool avgPath, bool mapFlag, bool minPrune) 
{
	// Check if depth is 0, if it is, then compute the score using getScore()
	// Traverse over the possible moves and create the new game state after performing each move
	// Recursively call the function minimax for each state, store the min or max
	// Alpha beta pruning: Also pass variables alpha, beta for pruning.
	// Fooling heuristic (Probably not needed): Choose alternate moves (Only do this in intermediate stages)
	// Finally check for the minimum or maximum value and modify the optMove variable to store the best move.

	if ((totCount > 5000) || ((flag == 2) && ((alpha > beta) || (abs(alpha - beta) < 0.75)))) {
		if (gameBoard.turn == ourTurn) {
			return 1000000;
		}
		else {
			return -1000000;
		}
	}

	if (mapFlag) {
		double storeVal = getStoredState(gameBoard);
		if (storeVal > -100000) {
			return storeVal;
		}
	}

	if (minPrune) {
		double tmpScore = getBoardScore(gameBoard, risk, ourTurn);
		// cout << tmpScore << " with currVal " << currVal << endl;
		if (tmpScore < currVal * (0.8)) {
			if (gameBoard.turn == ourTurn) {
				return 1000000;
			}
			else {
				return -1000000;
			}
		}
	}

	totCount++;
	double optVal;
	index optMoveNew;
	// Stores the optimal value at this node

	if (depth <= 0) {
		double score = getBoardScore(gameBoard, risk, ourTurn);
		// cout << "Getting new state! " << totCount << " with score : " << score << " and currVal : " << currVal << "\n";
		// cout << alpha << " and " << beta << endl;
		// printGameState(gameBoard, risk, ourTurn);
		return score;
	}
	else {
		int i, j;
		vector <reversiGame> gameList;
		vector <double> scoreList;

		// printGameState(gameBoard, risk, ourTurn);
		// cout << "The candList here is\n";
		// for (int i = 0; i < gameBoard.candList.size(); i++) {
		// 	cout << gameBoard.candList[i].x << " with " << gameBoard.candList[i].y << endl;
		// }

		// cout << "The possible moves for " << gameBoard.turn << " are\n";
		for (i = 0; i < gameBoard.candList.size(); i++) {
			// cout << gameBoard.candList[i].x << " and " << gameBoard.candList[i].y << endl;
			// cout << "MOVE BEFORE\n";
			// printGameState(gameBoard, risk, ourTurn);
			reversiGame tmpGame = getNewState(gameBoard, gameBoard.candList[i]);
			// printGameState(tmpGame, risk, ourTurn);
			gameList.push_back(tmpGame);
		}

		// if (gameBoard.candList.size() == 0) {
		// 	reversiGame tmpGame = getNewState(gameBoard, createIndex(-1, -1), 1);
		// 	gameList.push_back(tmpGame);
		// }

		// cout << "ENDING GENERATING POSSIBLE MOVES\n";

		if (flag == 0) {
			for (i = 0; i < gameList.size(); i++) {
				double tmpScore = minimax(gameList[i], risk, ourTurn, row, col, depth - 1, optMoveNew, alpha, beta, flag, avgPath, mapFlag, minPrune);
				scoreList.push_back(tmpScore);
			}
			if (gameBoard.turn == ourTurn) {
				// This is a max node, thus we maximize our score
				optVal = -10000;
				if (scoreList.size())
					optMove = gameBoard.candList[0];
				for (i = 0; i < scoreList.size(); i++) {
					if (optVal < scoreList[i]) {
						optVal = scoreList[i];
						optMove = gameBoard.candList[i];
						// cout << optVal << "|";
					}
				}
				if (optVal <= -10000) {
					optVal = getBoardScore(gameBoard, risk, ourTurn);
				}
			}
			else {
				// This is a min node, thus they try to minimize our score
				optVal = 100000;
				if (scoreList.size())
					optMove = gameBoard.candList[0];
				for (i = 0; i < scoreList.size(); i++) {
					if (optVal > scoreList[i]) {
						optVal = scoreList[i];
						optMove = gameBoard.candList[i];
					}
				}
				if (optVal >= 10000) {
					optVal = getBoardScore(gameBoard, risk, ourTurn);
				}
			}
		}
		else if (flag == 1) {
			for (i = 0; i < gameList.size(); i++) {
				double tmpScore = minimax(gameList[i], risk, ourTurn, row, col, depth - 1, optMoveNew, alpha, beta, flag, avgPath, mapFlag, minPrune);
				if (gameBoard.turn == ourTurn)
					tmpScore += getDynamicRisk(gameBoard, risk, gameBoard.candList[i].x, gameBoard.candList[i].y)*(0.1);
				scoreList.push_back(tmpScore);
			}
			if (gameBoard.turn == ourTurn) {
				// This is a max node, thus we maximize our score
				optVal = -100000;
				for (i = 0; i < scoreList.size(); i++) {
					if (optVal < scoreList[i]) {
						optVal = scoreList[i];
						optMove = gameBoard.candList[i];
					}
				}
				sort(scoreList.begin(), scoreList.end());
				optVal = 0;
				int cnt = 0;
				for (i = scoreList.size()-1; i >= max(0, (int) scoreList.size() - 3); i--) {
					optVal += scoreList[i];
					cnt++;
				}
				if (cnt)
					optVal /= cnt;
				if (scoreList.size() == 0) {
					optVal = getBoardScore(gameBoard, risk, ourTurn);
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
				sort(scoreList.begin(), scoreList.end());
				optVal = 0;
				int cnt = 0;
				for (i = 0; i < min(3, (int) scoreList.size()); i++) {
					optVal += scoreList[i];
					cnt++;
				}
				if (cnt)
					optVal /= cnt;
				if (scoreList.size() == 0) {
					optVal = getBoardScore(gameBoard, risk, ourTurn);
				}
			}
		}
		else {
			for (i = 0; i < gameList.size(); i++) {
				double tmpScore = minimax(gameList[i], risk, ourTurn, row, col, depth - 1, optMoveNew, alpha, beta, flag, avgPath, mapFlag, minPrune);
				if (gameBoard.turn == ourTurn)
					tmpScore += getDynamicRisk(gameBoard, risk, gameBoard.candList[i].x, gameBoard.candList[i].y)*(0.2);
				scoreList.push_back(tmpScore);
				if (gameBoard.turn == ourTurn) {
					if (alpha < tmpScore) {
						alpha = tmpScore;
						optMove = gameBoard.candList[i];
					}
				}
				else {
					if (beta > tmpScore) {
						beta = tmpScore;
						optMove = gameBoard.candList[i];
					}
				}
			}
			if (gameBoard.turn == ourTurn) 
				optVal = alpha;
			else
				optVal = beta;

			if (scoreList.size() == 0) {
				optVal = getBoardScore(gameBoard, risk, ourTurn);
			}

			if (scoreList.size())
				optMove = gameBoard.candList[0];
			for (i = 0; i < scoreList.size(); i++) {
				if (abs(optVal - scoreList[i]) < 0.0001)
					optMove = gameBoard.candList[i];
			}
		}
	}

	if (avgPath)
		optVal = getBoardScore(gameBoard, risk, ourTurn)*(0.6) + (optVal * 0.4);

	// printGameState(gameBoard, risk, ourTurn);
	// cout << alpha << " and " << beta << " with " << optVal <<endl;

	if (mapFlag) {
		string key = getBoardString(gameBoard);
		stateMap.insert(make_pair(key, optVal));
	}

	return optVal;
}

// Gives the number of opponents pieces turned if positioned here (at ind) by player turn
int getTurnOver(reversiGame gameBoard, index ind, int turn) {
	int ans = 0, i, j, tmpVal = 0;
	
	// Horizontal and vertical gains

	i = ind.x, j = ind.y;
	tmpVal = 0, j++;
	while ((j < gameBoard.col) && (gameBoard.board[i][j] == (3 - turn))) {
		j++;
		tmpVal++;
	}
	if ((j < gameBoard.col) && (gameBoard.board[i][j] == turn)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, j--;
	while ((j >= 0) && (gameBoard.board[i][j] == (3 - turn))) {
		j--;
		tmpVal++;
	}
	if ((j >= 0) && (gameBoard.board[i][j] == turn)) {
		ans += tmpVal;
	}
	
	i = ind.x, j = ind.y;
	tmpVal = 0, i++;
	while ((i < gameBoard.row) && (gameBoard.board[i][j] == (3 - turn))) {
		i++;
		tmpVal++;
	}
	if ((i < gameBoard.row) && (gameBoard.board[i][j] == turn)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, i--;
	while ((i >= 0) && (gameBoard.board[i][j] == (3 - turn))) {
		i--;
		tmpVal++;
	}
	if ((i >= 0) && (gameBoard.board[i][j] == turn)) {
		ans += tmpVal;
	}

	// Diagonal gains

	i = ind.x, j = ind.y;
	tmpVal = 0, j++, i++;
	while ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - turn))) {
		j++;
		i++;
		tmpVal++;
	}
	if ((j < gameBoard.col) && (i < gameBoard.row) && (gameBoard.board[i][j] == turn)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, j--, i++;
	while ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == (3 - turn))) {
		j--;
		i++;
		tmpVal++;
	}
	if ((j >= 0) && (i < gameBoard.row) && (gameBoard.board[i][j] == turn)) {
		ans += tmpVal;
	}
	
	i = ind.x, j = ind.y;
	tmpVal = 0, i--, j++;
	while ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == (3 - turn))) {
		i--;
		j++;
		tmpVal++;
	}
	if ((i >= 0) && (j < gameBoard.col) && (gameBoard.board[i][j] == turn)) {
		ans += tmpVal;
	}

	i = ind.x, j = ind.y;
	tmpVal = 0, i--, j--;
	while ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == (3 - turn))) {
		i--;
		j--;
		tmpVal++;
	}
	if ((i >= 0) && (j >= 0) && (gameBoard.board[i][j] == turn)) {
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

	int stat[10][10] = {
		500,-30,40,10,10,10,10,40,-30,500,
		-30,-80,-2,-2,-2,-2,-2,-2,-80,-30,
		40,-2,-1,-1,-1,-1,-1,-1,-2,40,
		10,-2,-1,-1,-1,-1,-1,-1,-2,10,
		10,-2,-1,-1,-1,-1,-1,-1,-2,10,
		10,-2,-1,-1,-1,-1,-1,-1,-2,10,
		10,-2,-1,-1,-1,-1,-1,-1,-2,10,
		40,-2,-1,-1,-1,-1,-1,-1,-2,40,
		-30,-80,-2,-2,-2,-2,-2,-2,-80,-30,
		500,-30,40,10,10,10,10,40,-30,500
	};

	for (i = 0; i < gameBoard.row; i++)
		for (j = 0; j < gameBoard.col; j++)
			risk[i][j] = stat[i][j];

	// for (i = 0; i < gameBoard.row; i++) {
	// 	for (j = 0; j < gameBoard.col; j++) {
	// 		if ((j == 0) || (j == gameBoard.col - 1)) {
	// 			if ((i == 0) || (i == gameBoard.row - 1)) {
	// 				risk[i][j] = 50;
	// 			}
	// 			else if ((i == 1) || (i == gameBoard.row - 2)) {
	// 				risk[i][j] = 0;
	// 			}
	// 			else {
	// 				risk[i][j] = 5;
	// 			}
	// 		} 
	// 		else if ((j == 1) || (j == gameBoard.col - 2)) {
	// 			if ((i < 2) || (i > gameBoard.row - 3)) {
	// 				risk[i][j] = 0;
	// 			}
	// 			else {
	// 				risk[i][j] = 0;
	// 			}
	// 		} 
	// 		else {
	// 			if ((i == 0) || (i == gameBoard.row - 1)) {
	// 				risk[i][j] = 5;
	// 			}
	// 			else if ((i == 1) || (i == gameBoard.row - 2)) {
	// 				risk[i][j] = 0;
	// 			}
	// 			else {
	// 				risk[i][j] = 0;
	// 			}
	// 		}
	// 	}
	// }
}

// Gives the optimal move for the current board status
// Method A : Greedy Strategy using a weighted sum of turnOvers and riskRegions
index getOptimalMoveMethodA(reversiGame& gameBoard, double tradeOff) {
	boardGame risk;
	getRiskRegions(gameBoard, risk);	
	index optInd{0, 0}, tmpInd;
	int i, j, a, b;
	double maxVal = 0, tmpVal;
	for (i = 0; i < gameBoard.candList.size(); i++) {
		tmpInd = gameBoard.candList[i];
		a = getTurnOver(gameBoard, tmpInd, gameBoard.turn);
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
	boardGame risk;
	double alpha = -1000000, beta = 1000000;
	getRiskRegions(gameBoard, risk);	
	stateMap.clear();
	currVal = getBoardScore(gameBoard, risk, gameBoard.turn);
	// cout << currVal << endl;

	// cout <<	minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 5, optInd) << endl;
	// cout <<	minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 5, optInd, alpha, beta, 2, true) << endl;
	if ((gameBoard.cntUs + gameBoard.cntOpp) < 15) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 5, optInd, alpha, beta, 2, true, true, false);
	}
	else if (currVal < 40) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 4, optInd, alpha, beta, 2, true, true, false);
	}
	else if ((gameBoard.candList.size() < 5) && ((gameBoard.cntUs + gameBoard.cntOpp) > 90)) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 8, optInd, alpha, beta, 2, true, true, false);
	}
 	else if ((gameBoard.cntUs + gameBoard.cntOpp) > 90) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 8, optInd, alpha, beta, 2, true, true, false);
	}
	else if ((gameBoard.candList.size() < 3) && ((gameBoard.cntUs + gameBoard.cntOpp) > 80)) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 7, optInd, alpha, beta, 2, true, true, true);
	}
	else if ((gameBoard.candList.size() < 5) && ((gameBoard.cntUs + gameBoard.cntOpp) > 80)) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 6, optInd, alpha, beta, 2, true, true, true);
	}
	else if ((gameBoard.candList.size() < 5) && ((gameBoard.cntUs + gameBoard.cntOpp) > 70)) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 5, optInd, alpha, beta, 2, true, true, true);
	}
	else if ((gameBoard.candList.size() < 9) && ((gameBoard.cntUs + gameBoard.cntOpp) > 70)) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 5, optInd, alpha, beta, 2, true, true, true);
	}
	else if ((gameBoard.candList.size() == 2) && ((gameBoard.cntUs + gameBoard.cntOpp) > 50)) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 5, optInd, alpha, beta, 2, true, true, true);
	}
	else if ((gameBoard.candList.size() == 1) && ((gameBoard.cntUs + gameBoard.cntOpp) > 50)) {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 5, optInd, alpha, beta, 2, true, true, true);
	}	
	else {
		minimax(gameBoard, risk, gameBoard.turn, gameBoard.row, gameBoard.col, 3, optInd, alpha, beta, 2, true, true, false);
	}

	return optInd;
}
