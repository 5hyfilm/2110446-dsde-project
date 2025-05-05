CREATE TABLE IF NOT EXISTS issues (
    id SERIAL PRIMARY KEY,
    message_id INT UNIQUE NOT NULL,
    type VARCHAR(50),
    coordinates GEOGRAPHY(Point, 4326),
    problem_type_fondue TEXT[], 
    org TEXT[],              
    org_action TEXT[],       
    description TEXT,
    photo_url TEXT,
    address TEXT,
    subdistrict VARCHAR(255),
    district VARCHAR(255),
    province VARCHAR(255),
    timestamp TIMESTAMP,
    state VARCHAR(50),
    last_activity TIMESTAMP
);
