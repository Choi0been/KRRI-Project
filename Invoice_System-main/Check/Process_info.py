import pymysql
from jamo import hangul_to_jamo

# 테이블 매핑 정의
table_mappings = {
    "SIDO_INFO": ["address_do"],
    "SIGUNGU_INFO": ["address_si", "address_gun", "address_gu"],
    "EUPMYEONDONG_INFO": ["address_eup", "address_myeon", "address_dong"],
    "RI_INFO": ["address_ri"],
    "ROAD_INFO": ["address_ro_name", "address_gil_name"]
}

# 외래키 컬럼 정의
foreign_key_columns = {
    "SIDO_INFO": [],
    "SIGUNGU_INFO": ["SIDO_ID"],
    "EUPMYEONDONG_INFO": ["SIGUNGU_ID"],
    "RI_INFO": ["EUPMYEONDONG_ID"],
    "ROAD_INFO": ["SIGUNGU_ID"]
}

# MySQL 연결해보기
def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='0000',
        db='ADDR_INFO_AREA',
        charset='utf8'
    )

# 편집 거리 계산 (초성-중성-종성 분해 후 측정!)
def jamo_levenshtein(s1, s2):
    s1 = ''.join(hangul_to_jamo(s1))
    s2 = ''.join(hangul_to_jamo(s2))
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[len_s1][len_s2]

# DB에서 후보 목록 가져오기
def fetch_candidates(table, fk_conditions):
    conn = get_connection()
    cursor = conn.cursor()

    query = f"SELECT {table[:-5]}_ID, {table[:-5]}_NAME FROM {table}"
    where_clauses = []
    fk_cols = foreign_key_columns.get(table, [])

    for col in fk_cols:
        if col in fk_conditions:
            where_clauses.append(f"{col} = '{fk_conditions[col]}'")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    print(f"[QUERY] {query}")
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    print(f"[DB] {table}: {len(results)}개 후보 로드됨")
    return results

# 후보 중 가장 유사한 항목 선택
def best_match(token, candidates):
    min_dist = float('inf')
    best = token, None
    for cand_id, cand_name in candidates:
        dist = jamo_levenshtein(token, cand_name)
        if dist < min_dist:
            min_dist = dist
            best = (cand_name, cand_id)
    print(f"[MATCH] '{token}' → '{best[0]}' (ID: {best[1]}, 거리: {min_dist})")
    return best

# 주소 보정 수행
def correct_text_label_pairs(text_label_pairs):
    corrected = []
    fk_conditions = {}

    for token, token_class in text_label_pairs:
        current_table = None
        for table_name, class_list in table_mappings.items():
            if any(token_class.endswith(c) for c in class_list):
                current_table = table_name
                break

        if not current_table:
            corrected.append(token)
            continue

        candidates = fetch_candidates(current_table, fk_conditions)
        best_name, best_id = best_match(token, candidates)
        corrected.append(best_name if best_name else token)

        id_col_name = f"{current_table[:-5]}_ID"
        fk_conditions[id_col_name] = best_id

    return corrected
