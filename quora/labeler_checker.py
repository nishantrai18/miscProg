#!/usr/bin/env python

import math
import sys


TOP_TOPICS = [
    183,
    29,
    189,
    78,
    159,
    142,
    147,
    190,
    45,
    124,
]


def get_question_score(expected_line, output_line, default_guesses):
    expected = set(map(int, expected_line.strip().split()[1:]))
    actual_toks = output_line.strip().split()

    score = 0.0
    max_score = 0.0
    min_score = 0.0
    for i in range(min(len(actual_toks), 10)):
        idx_score = math.sqrt(10.0 - i)
        if i < len(expected):
            max_score += idx_score
        if default_guesses[i] in expected:
            min_score += idx_score
        try:
            guess = int(actual_toks[i])
            if guess in expected:
                score += idx_score
        except:
            # not an integer, w/e
            pass

    # print >> sys.stderr, min_score, score, max_score
    return (score / max_score, min_score / max_score)


def check(expected, output):
    expectedlines = expected.splitlines()
    outputlines = output.splitlines()

    if len(outputlines) < len(expectedlines):
        print >> sys.stderr, 'Got fewer lines than expected'
        return 0.0

    max_score = len(expectedlines)
    score = 0.0
    min_score = 0.0
    for i in range(len(expectedlines)):
        question_score, question_min_score = get_question_score(expectedlines[i], outputlines[i], TOP_TOPICS)
        score += question_score
        min_score += question_min_score

    normalized_score = max(0.0, (score - min_score) / (max_score - min_score))

    print >> sys.stderr, 'Min score:', min_score
    print >> sys.stderr, 'Max score:', max_score
    print >> sys.stderr, 'Your raw score:', score
    print >> sys.stderr, 'Your normalized score:', normalized_score

    return normalized_score


def main():
    check(open(sys.argv[1]).read(), open(sys.argv[2]).read())


if __name__ == '__main__':
    main()
