import asyncio
import sys
import os

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.mongodb import courses_list_collection

COURSE_CODES = [
  "BUS/B2/WD/478", "BUS/B1/WD/795", "Eve/A1/WD/467", "ENG/B2/WD/863", "ENG/B2/WD/183",
  "ENG/B1/WD/914", "ENG/A2/WD/978", "ENG/A1/WD/533", "ENG/B2/WD/345", "ENG/B1/WD/527",
  "ENG/A2/WD/802", "ENG/A1/WD/211", "ENG/A2/WE/946", "ENG/A1/WE/809", "ENG/B1/WE/203",
  "ENG/B2/WE/352", "ENG/B1/WE/569", "ENG/A2/WE/378", "ENG/A1/WE/285", "ENG/B2/WE/914",
  "ENG/B1/WE/517", "GMA/WD/291", "GMA/WD/920", "GMA/WD/722", "GMA/B2/WE/901",
  "GMA/WE/193", "GMA/WE/874", "Eve/HSK1/WD/650", "CHI/HSK1/WD/161", "CHI/HSK2/WD/750",
  "CHI/HSK1/WD/278", "CHI/HSK6/WE/770", "CHI/C1/WE/459", "CHI/B2/WE/176", "CHI/B1/WE/822",
  "CHI/A2/WE/854", "CHI/A1/WE/676", "GRE/WD/917", "GRE/WD/643", "GRE/WD/551",
  "GRE/WE/555", "GRE/WE/327", "GRE/B2/WE/864", "GRE/A1/WE/325", "GRE/B1/WE/789",
  "GRE/A2/WE/403", "GRE/B2/WE/633", "GRE/C1/WE/848", "GRE/C2/WE/438", "POR/C2/WD/459",
  "POR/C1/WD/997", "POR/B2/WD/540", "POR/B1/WD/183", "POR/A2/WD/758", "POR/A1/WD/862",
  "POR/C2/WE/391", "POR/C1/WE/209", "POR/B2/WE/424", "POR/B1/WE/798", "POR/A2/WE/916",
  "POR/A1/WE/108", "FRE/A1/WD/177", "FRE/A2/WD/135", "FRE/B1/WD/312", "FRE/B2/WD/801",
  "FRE/A1/WD/943", "FRE/A2/WD/880", "FRE/B1/WD/987", "FRE/B2/WD/434", "Eve/A1/WD/941",
  "Eve/A1/WD/120", "FRE/B2/WE/468", "FRE/B1/WE/442", "FRE/A2/WE/543", "FRE/A1/WE/601",
  "FRE/B2/WE/276", "FRE/B1/WE/641", "FRE/A2/WE/702", "FRE/A1/WE/315", "GER/A1/WD/899",
  "GER/A2/WD/818", "GER/B1/WD/154", "GER/B2/WD/888", "GER/A1/WD/350", "GER/A2/WD/109",
  "GER/B1/WD/219", "GER/B2/WD/192", "Eve/A1/WD/135", "Eve/A2/WD/136", "Eve/B1/WD/805",
  "Eve/B2/WD/553", "GER/A1/WD/599", "GER/A2/WD/304", "GER/B1/WD/967", "GER/B2/WD/571",
  "Eve/A1/WD/286", "GER/B2/WE/871", "GER/B1/WE/545", "GER/A2/WE/639", "GER/A1/WE/234",
  "GER/B2/WE/280", "GER/B1/WE/161", "GER/A2/WE/442", "GER/A1/WE/937", "GER/B2/WE/363",
  "GER/B1/WE/369", "GER/A2/WE/125", "GER/A1/WE/961", "HAU/B2/WD/450", "HAU/B1/WD/618",
  "HAU/A2/WD/248", "YOR/A2/WD/366", "YOR/A1/WD/260", "Yor/A1/WE/237", "ITA/C2/WD/399",
  "ITA/C1/WD/945", "ITA/B2/WD/881", "ITA/B1/WD/293", "ITA/A2/WD/257", "ITA/A1/WD/409",
  "ITA/C2/WE/725", "ITA/C1/WE/142", "ITA/B2/WE/127", "ITA/B1/WE/366", "ITA/A2/WE/659",
  "ITA/A1/WE/411", "SPA/A1/WE/483", "SPA/A2/WE/935", "SPA/B1/WE/452", "SPA/B2/WE/830",
  "SPA/B2/WD/970", "SPA/B1/WD/236", "SPA/A2/WD/707", "SPA/A1/WD/927", "SPA/B2/WD/537",
  "SPA/B1/WD/357", "SPA/A2/WD/864", "SPA/A1/WD/155", "IEL/B2/WD/563", "IEL/B2/WD/427",
  "Eve/B2/WD/662", "IEL/WE/193", "IEL/WE/320", "IEL/WE/208", "TOE/B2/WD/814",
  "HAU/A1/WD/842", "SAT/WD/462", "SAT/WD/538", "SAT/WD/894", "PTE/B2/WD/237",
  "TUR/A1/WD/847", "TUR/C2/WE/246", "TUR/C1/WE/712", "TUR/B2/WE/439", "TUR/A2/WE/755",
  "TUR/A1/WE/752", "TUR/A1/WE/481", "CEL/B2/WD/330", "CEL/B2/WE/223", "Eve/B2/WD/238",
  "CEL/B2/WD/306", "RUS/A1/WD/412", "RUS/A1/WD/549", "RUS/B1/WD/449", "RUS/B2/WD/162",
  "RUS/C1/WD/782", "RUS/C2/WD/585", "RUS/A1/WE/637", "RUS/B1/WE/879", "RUS/B2/WE/906",
  "RUS/C1/WE/298", "RUS/C2/WE/338", "RUS/A1/WE/639", "IGB/A2/WD/877", "IGB/A1/WD/292",
  "HAU/A1/WE/872", "OET/B2/WD/392", "OET/B2/WE/760", "SEL/B2/WD/131", "ESO/B2/WD/284",
  "Fre/A1/WD/440", "Duo/B2/WD/812", "Fin/A1/WE/795", "Dut/A1/WE/346", "Fin/A2/WE/238",
  "TUR/B1/WE/443", "HEB/A1/WE/496", "HEB/A2/WE/767", "HEB/B1/WE/342", "HEB/B2/WE/799",
  "HEB/C2/WE/602", "HEB/C1/WE/196", "POL/A2/WE/410", "POL/A1/WE/595", "POL/B1/WE/774",
  "POL/B2/WE/672", "POL/C1/WE/169", "POL/C2/WE/609", "KOR/A1/WE/139", "KOR/A2/WE/337",
  "KOR/B1/WE/178", "KOR/B2/WE/482", "KOR/C1/WE/402", "KOR/C2/WE/788", "JAP/C1/WE/524",
  "JAP/B2/WE/779", "JAP/B1/WE/256", "JAP/A2/WE/661", "JAP/A1/WE/245", "HIN/A2/WE/255",
  "HIN/A1/WE/451", "HIN/B1/WE/803", "HIN/B2/WE/908", "HIN/C1/WE/385", "HIN/C2/WE/805"
]

SUBJECT_MAPPING = {
  "FRE": "French Language",
  "GER": "German Language",
  "SPA": "Spanish Language",
  "ENG": "English Language",
  "CHI": "Chinese (HSK)",
  "POR": "Portuguese Language",
  "ITA": "Italian Language",
  "RUS": "Russian Language",
  "JAP": "Japanese Language",
  "KOR": "Korean Language",
  "YOR": "Yoruba Language",
  "HAU": "Hausa Language",
  "IGB": "Igbo Language",
  "GRE": "GRE",
  "IEL": "IELTS",
  "TOE": "TOEFL",
  "SAT": "SAT",
  "TUR": "Turkish Language",
  "HEB": "Hebrew Language",
  "POL": "Polish Language",
  "HIN": "Hindi Language",
  "FIN": "Finnish Language",
  "DUT": "Dutch Language",
  "BUS": "Business English",
  "EVE": "Evening Classes",
  "OET": "OET Prep",
  "SEL": "SELT",
  "ESO": "ESOL",
  "DUO": "Duolingo",
  "PTE": "PTE",
  "CEL": "CELPIP",
  "GMA": "GMAT"
}

def get_subject_name(code):
    prefix = code.split('/')[0].upper()
    return SUBJECT_MAPPING.get(prefix, "General / Other")

async def seed_courses():
    print(f"Starting seeding of {len(COURSE_CODES)} default courses...")
    seeded_count = 0
    for idx, code in enumerate(COURSE_CODES):
        subject = get_subject_name(code)
        parts = code.split('/')
        level = parts[1] if len(parts) > 1 else "General"
        
        course_name = f"{subject} ({level})"
        
        # Upsert: only update if not exists, to preserve any edits teachers make
        result = await courses_list_collection.update_one(
            {"course_code": code},
            {
                "$setOnInsert": {
                    "course_code": code,
                    "course_name": course_name,
                    "level": level,
                    "category": subject,
                }
            },
            upsert=True
        )
        if result.upserted_id:
            seeded_count += 1

    print(f"Seeding completed! Seeded {seeded_count} new courses.")

if __name__ == "__main__":
    asyncio.run(seed_courses())
