package salary.db

import com.mongodb.casbah.Imports._

class DatabaseManager {

  val mongoClient = MongoClient()
  val salaryDB = mongoClient("salary")

}
